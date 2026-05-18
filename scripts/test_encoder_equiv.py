"""
Equivalence test: vectorized ASTEncoder.forward vs the golden
_forward_reference per-node oracle.

Builds ~50 diverse random ASTs (Add/Mul/Pow/nested/deep/leaf-only/large,
including POW-heavy and depth>=6 cases to catch child-order bugs), runs the
encoder in eval mode, and asserts torch.allclose on BOTH outputs at
atol=1e-5, rtol=1e-4.

Run:
    cd C:\\GitHub\\ISRE
    set PYTHONPATH=.
    set PYTHONIOENCODING=utf-8
    python scripts/test_encoder_equiv.py
"""

import random
import sys

import torch

from isre.learning.encoder import ASTEncoder
from isre.symbolic.isre_ast import Num, Var, Add, Mul, Pow

ATOL = 1e-5
RTOL = 1e-4


def rand_leaf(rng: random.Random):
    if rng.random() < 0.6:
        return Num(rng.randint(-9, 9))
    return Var(rng.choice(["x", "y", "z", "w"]))


def rand_ast(rng: random.Random, depth: int):
    """Recursively build a random AST up to `depth`."""
    if depth <= 0 or rng.random() < 0.3:
        return rand_leaf(rng)
    kind = rng.choice(["add", "mul", "pow", "add", "mul"])
    if kind == "pow":
        base = rand_ast(rng, depth - 1)
        exp = Num(rng.randint(0, 4))
        return Pow(base, exp)
    n_children = rng.randint(2, 4)
    children = [rand_ast(rng, depth - 1) for _ in range(n_children)]
    return Add(*children) if kind == "add" else Mul(*children)


def build_corpus():
    rng = random.Random(1234)
    asts = []

    # Hand-crafted edge cases.
    asts.append(Var())                                  # leaf-only
    asts.append(Num(7))                                 # leaf-only number
    asts.append(Add(Var(), Num(1)))                     # simple
    asts.append(Mul(Num(3), Pow(Var(), Num(2))))        # pow inside mul
    asts.append(Add(
        Mul(Num(3), Pow(Var(), Num(2))),
        Mul(Num(2), Var()),
        Num(-5),
    ))
    # POW-heavy, nested non-commutative (child-order sensitive).
    asts.append(Pow(Pow(Pow(Var(), Num(2)), Num(3)), Num(4)))
    asts.append(Pow(Add(Var(), Num(1)), Num(3)))
    asts.append(Mul(
        Pow(Var("x"), Num(2)),
        Pow(Var("y"), Num(3)),
        Pow(Var("z"), Num(4)),
    ))
    # Deep chains (depth >= 6).
    deep = Var()
    for _ in range(8):
        deep = Pow(deep, Num(2))
    asts.append(deep)
    deep2 = Num(1)
    for i in range(7):
        deep2 = Add(deep2, Mul(Num(i + 1), Var()))
    asts.append(deep2)

    # Random diverse ASTs at varied depths (include depth >= 6).
    for _ in range(40):
        depth = rng.randint(2, 8)
        asts.append(rand_ast(rng, depth))

    return asts


def main():
    torch.manual_seed(0)
    encoder = ASTEncoder(hidden_dim=128, num_rounds=6)
    encoder.eval()

    asts = build_corpus()

    max_emb_diff = 0.0
    max_ctx_diff = 0.0
    failures = []

    for idx, ast in enumerate(asts):
        with torch.no_grad():
            emb_v, ctx_v = encoder.forward(ast)
            emb_r, ctx_r = encoder._forward_reference(ast)

        emb_diff = (emb_v - emb_r).abs().max().item()
        ctx_diff = (ctx_v - ctx_r).abs().max().item()
        max_emb_diff = max(max_emb_diff, emb_diff)
        max_ctx_diff = max(max_ctx_diff, ctx_diff)

        emb_ok = torch.allclose(emb_v, emb_r, atol=ATOL, rtol=RTOL)
        ctx_ok = torch.allclose(ctx_v, ctx_r, atol=ATOL, rtol=RTOL)
        if not (emb_ok and ctx_ok):
            failures.append((idx, ast, emb_diff, ctx_diff))

    print(f"Tested {len(asts)} ASTs in eval mode.")
    print(f"max abs diff  node_embeddings: {max_emb_diff:.3e}")
    print(f"max abs diff  global_context:  {max_ctx_diff:.3e}")

    if failures:
        print(f"\nFAILED: {len(failures)} AST(s) exceeded atol={ATOL}, rtol={RTOL}")
        for idx, ast, ed, cd in failures[:10]:
            print(f"  AST[{idx}] expr={ast.to_expr()}")
            print(f"    node_embeddings max diff = {ed:.3e}")
            print(f"    global_context  max diff = {cd:.3e}")
        sys.exit(1)

    print("\nPASS: all ASTs within tolerance.")
    sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""trajectory_viewer.py - Generate static HTML pages for trajectory inspection."""
from __future__ import annotations
import argparse, html, json, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from jinja2 import Environment, DictLoader, ChoiceLoader, FileSystemLoader, select_autoescape
except ImportError:
    print("jinja2 not installed"); sys.exit(1)

try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

DEFAULT_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{{ trajectory_id }}</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--panel-2:#1d2230;--text:#e8ecf3;--muted:#9aa4b2;--line:#2a3242;--gold:#3b2f12;--gold-border:#b58a2a;--top:#143125;--top-border:#2ea568;--accent:#6aa7ff;--bad:#ff6b6b;--good:#50c878;--code:#11151d}
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,sans-serif;background:var(--bg);color:var(--text);line-height:1.45}
.wrap{max-width:1440px;margin:0 auto;padding:24px}
.header,.step-card{background:var(--panel);border:1px solid var(--line);border-radius:16px;box-shadow:0 6px 24px rgba(0,0,0,.18)}
.header{padding:20px;margin-bottom:20px}.header h1{margin:0 0 8px;font-size:28px}
.sub{color:var(--muted);font-size:14px;margin-bottom:8px;word-break:break-word}
.pill-row{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.pill{padding:6px 10px;border-radius:999px;background:var(--panel-2);border:1px solid var(--line);font-size:13px}
.verdict-success{color:var(--good);font-weight:700}.verdict-fail{color:var(--bad);font-weight:700}
.toolbar{display:flex;gap:10px;margin:16px 0 20px}
button{cursor:pointer;color:var(--text);background:var(--panel-2);border:1px solid var(--line);border-radius:10px;padding:8px 12px;font-size:14px}
button:hover{border-color:var(--accent)}
.step-card{margin-bottom:18px;overflow:hidden}
.step-head{padding:14px 16px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line)}
.step-title{font-size:18px;font-weight:700}
.step-body{padding:16px;display:grid;grid-template-columns:1.15fr .85fr;gap:16px}
@media(max-width:1000px){.step-body{grid-template-columns:1fr}}
.section{background:rgba(255,255,255,.015);border:1px solid var(--line);border-radius:14px;padding:14px}
.section h3{margin:0 0 10px;font-size:16px}
.expr{font-family:ui-monospace,Consolas,monospace;background:var(--code);border:1px solid var(--line);border-radius:10px;padding:10px 12px;overflow-x:auto;white-space:pre-wrap;word-break:break-word}
.diff-box{margin-top:10px;padding:10px 12px;border-radius:10px;background:rgba(164,86,214,.1);border:1px solid rgba(164,86,214,.35);font-size:13px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-top:10px}
.metric{background:var(--panel-2);border:1px solid var(--line);border-radius:10px;padding:10px}
.metric .k{color:var(--muted);font-size:12px;margin-bottom:4px}.metric .v{font-weight:700;font-size:15px}
.svg-wrap{width:100%;overflow-x:auto;background:#fff;border-radius:12px;padding:8px;border:1px solid var(--line)}
.svg-wrap pre{margin:0;color:#111;background:#fff;padding:8px;font-family:ui-monospace,Consolas,monospace;white-space:pre}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;border-bottom:1px solid var(--line);padding:8px 10px}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase}
tr.gold-row{background:var(--gold);outline:1px solid var(--gold-border)}
tr.top-row{background:var(--top);outline:1px solid var(--top-border)}
.small{color:var(--muted);font-size:12px}.collapsed .step-body{display:none}
.legend{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;font-size:12px;color:var(--muted)}
.legend span{display:inline-flex;gap:6px;align-items:center}
.dot{width:12px;height:12px;border-radius:999px;display:inline-block;border:1px solid var(--line)}
.dot.gold{background:#7a5a17;border-color:#b58a2a}.dot.top{background:#14522e;border-color:#2ea568}
.footer-note{margin-top:20px;color:var(--muted);font-size:12px}
</style></head><body><div class="wrap">
<div class="header"><h1>{{ trajectory_id }}</h1>
<div class="sub"><strong>Canonical:</strong> {{ canonical }}</div>
<div class="sub"><strong>Original:</strong> {{ original }}</div>
<div class="pill-row">
<div class="pill">Difficulty: {{ difficulty }}</div>
<div class="pill">Verdict: <span class="{{ 'verdict-success' if verdict.startswith('Success') else 'verdict-fail' }}">{{ verdict }}</span></div>
<div class="pill">Steps: {{ steps|length }}</div></div>
{% if inverse_sequence %}<div class="sub" style="margin-top:10px"><strong>Inverse sequence:</strong> {{ inverse_sequence }}</div>{% endif %}
<div class="legend"><span><i class="dot gold"></i> gold</span><span><i class="dot top"></i> model top</span></div></div>
<div class="toolbar"><button onclick="toggleAll(false)">Expand all</button><button onclick="toggleAll(true)">Collapse all</button></div>
{% for step in steps %}
<div class="step-card" id="step-{{ step.step_num }}">
<div class="step-head"><div><div class="step-title">Step {{ step.step_num }}</div><div class="small">Gold: {{ step.gold_action }}</div></div>
<button onclick="toggleOne('step-{{ step.step_num }}')">Toggle</button></div>
<div class="step-body"><div>
<div class="section"><h3>Expression</h3><div class="expr">{{ step.expr }}</div>
{% if step.diff_summary %}<div class="diff-box"><strong>Diff:</strong> {{ step.diff_summary }}</div>{% endif %}
<div class="metrics">{% for k, v in step.metrics.items() %}<div class="metric"><div class="k">{{ k }}</div><div class="v">{{ v }}</div></div>{% endfor %}</div></div>
<div class="section" style="margin-top:14px"><h3>Candidates</h3>
<table><thead><tr><th>#</th><th>Node</th><th>Action</th><th>Score</th><th>Rank</th><th>Flags</th></tr></thead><tbody>
{% for row in step.candidates_table %}
<tr class="{% if row.is_gold %}gold-row{% elif row.is_top %}top-row{% endif %}">
<td>{{ loop.index }}</td><td>{{ row.node_id }}</td><td>{{ row.action }}</td><td>{{ row.score }}</td><td>{{ row.rank }}</td>
<td>{% if row.is_gold %}gold{% endif %}{% if row.is_gold and row.is_top %} / {% endif %}{% if row.is_top %}top{% endif %}</td></tr>
{% endfor %}</tbody></table></div></div>
<div><div class="section"><h3>AST</h3><div class="svg-wrap">{{ step.ast_svg | safe }}</div></div></div>
</div></div>{% endfor %}
<div class="footer-note">Generated by trajectory_viewer.py</div></div>
<script>
function toggleOne(id){document.getElementById(id).classList.toggle('collapsed')}
function toggleAll(c){document.querySelectorAll('.step-card').forEach(e=>{if(c)e.classList.add('collapsed');else e.classList.remove('collapsed')})}
</script></body></html>"""

def safe_float(x):
    try: return float(x) if x is not None else None
    except: return None

def normalize_action_name(x):
    if isinstance(x, str): return x
    if isinstance(x, dict): return str(x.get("action", x.get("gold_action", x)))
    return str(x)

def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",",":"))

def ast_to_human_readable(d):
    def rec(d):
        t, v, ch = d["type"], d.get("value",""), d.get("children",[])
        if t in ("NUMBER","VARIABLE","CONST"): return str(v) if v != "" else t.lower()
        if t == "ADD": return " + ".join(f"({rec(c)})" if c["type"] in ("ADD","MUL") else rec(c) for c in ch)
        if t == "MUL": return " * ".join(f"({rec(c)})" if c["type"] in ("ADD","MUL") else rec(c) for c in ch)
        if t == "POW" and len(ch)==2: return f"{rec(ch[0])}^{rec(ch[1])}"
        return f"{t}({','.join(rec(c) for c in ch)})"
    try: return rec(d)
    except: return "[invalid AST]"

def ast_to_ascii(d):
    lines=[]
    def rec(d,prefix="",is_last=True):
        label=d.get("type","?")
        if d.get("value") not in ("",None): label+=f"({d['value']})"
        connector="" if not prefix else ("└─ " if is_last else "├─ ")
        lines.append(f"{prefix}{connector}{label}")
        ch=d.get("children",[])
        for i,c in enumerate(ch): rec(c,prefix+("   " if is_last else "│  "),i==len(ch)-1)
    rec(d); return "\n".join(lines)

def enumerate_ast_paths(d, path="0"):
    out={path:d}
    for i,c in enumerate(d.get("children",[])): out.update(enumerate_ast_paths(c,f"{path}.{i}"))
    return out

def diff_ast_paths(prev,curr):
    if prev is None: return set(),"No previous step"
    pm,cm=enumerate_ast_paths(prev),enumerate_ast_paths(curr)
    changed,added,removed,modified=set(),0,0,0
    for p in set(cm)-set(pm): changed.add(p); added+=1
    removed=len(set(pm)-set(cm))
    for p in set(cm)&set(pm):
        if canonical_json(cm[p])!=canonical_json(pm[p]): changed.add(p); modified+=1
    parts=[]
    if modified: parts.append(f"modified {modified}")
    if added: parts.append(f"added {added}")
    if removed: parts.append(f"removed {removed}")
    return changed, ", ".join(parts) or "no change"

def ast_to_svg_dot(d, highlight_paths=None, highlight_node_path=None):
    highlight_paths=highlight_paths or set()
    if not GRAPHVIZ_AVAILABLE:
        return "<pre>"+html.escape(ast_to_ascii(d))+"</pre>"
    dot=Digraph(format="svg"); dot.attr(rankdir="TB",nodesep="0.35",ranksep="0.55")
    dot.attr("node",fontname="Helvetica",fontsize="11"); dot.attr("edge",color="#666666")
    def add(d,path="0",parent=None):
        label=d.get("type","?")
        if d.get("value") not in ("",None): label+=f"\n{d['value']}"
        shape="box" if d.get("type","") in ("ADD","MUL","POW") else "ellipse"
        fill,col,pw="white","#999999","1.0"
        if path in highlight_paths: fill,col,pw="#ead7f7","#a456d6","2.0"
        if highlight_node_path and path==highlight_node_path: fill,col,pw="#d9ecff","#4b88d9","2.5"
        dot.node(path,label,shape=shape,style="filled",fillcolor=fill,color=col,penwidth=pw)
        if parent: dot.edge(parent,path)
        for i,c in enumerate(d.get("children",[])): add(c,f"{path}.{i}",path)
    add(d)
    try: return dot.pipe().decode("utf-8")
    except Exception as e: return f"<pre>{html.escape(str(e))}</pre><pre>{html.escape(ast_to_ascii(d))}</pre>"

def extract_candidates(step):
    raw=step.get("candidate_actions",step.get("candidates",[]))
    out=[]
    for item in raw:
        if isinstance(item,dict): out.append({"node_id":item.get("node_id","?"),"action":normalize_action_name(item.get("action",item))})
        elif isinstance(item,(list,tuple)) and len(item)>=2: out.append({"node_id":item[0],"action":normalize_action_name(item[1])})
        else: out.append({"node_id":step.get("gold_node_id","?"),"action":normalize_action_name(item)})
    return out

def compute_ranks(scores):
    if not scores or all(s is None for s in scores): return ["-"]*len(scores)
    idx=sorted([(i,s) for i,s in enumerate(scores) if s is not None],key=lambda x:(-x[1],x[0]))
    ranks=["-"]*len(scores); rank=1
    for pos,(i,_) in enumerate(idx):
        if pos>0 and idx[pos][1]<idx[pos-1][1]: rank=pos+1
        ranks[i]=str(rank)
    return ranks

def top_idx(scores):
    v=[(i,s) for i,s in enumerate(scores) if s is not None]
    return sorted(v,key=lambda x:(-x[1],x[0]))[0][0] if v else None

def load_trajectory(path):
    with path.open() as f: data=json.load(f)
    for s in data.get("steps",[]):
        if "state" in s and isinstance(s["state"],dict) and "state_dict" not in s: s["state_dict"]=s["state"]
    return data

def build_env():
    bl=DictLoader({"t.html.j2":DEFAULT_TEMPLATE})
    ext=Path(__file__).parent/"templates"
    loader=ChoiceLoader([FileSystemLoader(str(ext)),bl]) if ext.exists() else bl
    return Environment(loader=loader,autoescape=select_autoescape(["html","xml"]))

def render_trajectory(traj,pred,output_dir,source_path,env):
    template=env.get_template("t.html.j2")
    steps_html,prev=[], None
    for idx,step in enumerate(traj.get("steps",[])):
        sd=step.get("state_dict",step.get("state",{}))
        expr=step.get("state_expr") or step.get("expr") or ast_to_human_readable(sd)
        changed,diff_summary=diff_ast_paths(prev,sd)
        svg=ast_to_svg_dot(sd,highlight_paths=changed)
        cands=extract_candidates(step)
        gold_a=normalize_action_name(step.get("gold_action","?"))
        gold_nid=step.get("gold_node_id",step.get("applied_at_node_id","?"))
        sp=pred.get("steps",{}).get(str(idx)) if pred else None
        scores=[safe_float(x) for x in sp.get("scores",[])] if sp else []
        if len(scores)!=len(cands): scores=[None]*len(cands)
        ranks=compute_ranks(scores); ti=top_idx(scores)
        table=[]
        for i,c in enumerate(cands):
            ig=(c["action"]==gold_a and str(c["node_id"])==str(gold_nid))
            table.append({"node_id":c["node_id"],"action":c["action"],
                "score":f"{scores[i]:.4f}" if scores[i] is not None else "-",
                "rank":ranks[i],"is_gold":ig,"is_top":ti==i})
        rem=max(0,int(traj.get("difficulty",len(traj.get("steps",[]))))-idx-1)
        metrics={"complexity":step.get("complexity","?"),"remaining_steps":rem,
            "gold_rank":sp.get("gold_rank","?") if sp else "?",
            "changed_nodes":len(changed)}
        steps_html.append({"step_num":idx+1,"expr":expr,"ast_svg":svg,
            "candidates_table":table,"metrics":metrics,"gold_action":gold_a,
            "diff_summary":diff_summary if idx>0 else ""})
        prev=sd
    d=traj.get("difficulty",len(traj.get("steps",[])))
    reached=traj.get("reached_canonical",len(traj.get("steps",[]))==d)
    verdict="Success" if reached else f"Failed ({len(traj.get('steps',[]))}/{d})"
    ce=traj.get("canonical_expr") or (ast_to_human_readable(traj["canonical_ast"]) if "canonical_ast" in traj else "?")
    oe=traj.get("original_expr") or (steps_html[0]["expr"] if steps_html else "?")
    inv=traj.get("inverse_sequence",traj.get("inverse_transform_sequence",[]))
    h=template.render(trajectory_id=traj.get("trajectory_id",source_path.stem),
        canonical=ce,original=oe,difficulty=d,
        inverse_sequence=" → ".join(inv) if inv else "",steps=steps_html,verdict=verdict)
    op=output_dir/f"{traj.get('trajectory_id',source_path.stem)}.html"
    op.write_text(h,encoding="utf-8"); return op

def render_index(rows,output_dir):
    rh="".join(f'<tr><td><a href="{html.escape(r["fn"])}">{html.escape(r["id"])}</a></td>'
        f'<td>{r["d"]}</td><td class="{"good" if r["v"].startswith("Success") else "bad"}">{html.escape(r["v"])}</td>'
        f'<td>{html.escape(r["o"])}</td><td>{html.escape(r["c"])}</td></tr>' for r in rows)
    h=f"""<!doctype html><html><head><meta charset="utf-8"><title>Index</title>
<style>body{{font-family:Inter,system-ui,sans-serif;background:#0f1115;color:#e8ecf3;margin:0}}
.w{{max-width:1440px;margin:0 auto;padding:24px}}.card{{background:#171a21;border:1px solid #2a3242;border-radius:16px;padding:20px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #2a3242}}
th{{color:#9aa4b2;font-size:12px;text-transform:uppercase}}a{{color:#6aa7ff}}
.good{{color:#50c878;font-weight:700}}.bad{{color:#ff6b6b;font-weight:700}}</style></head>
<body><div class="w"><div class="card"><h1 style="margin-top:0">Index ({len(rows)} trajectories)</h1>
<table><thead><tr><th>ID</th><th>Diff</th><th>Verdict</th><th>Original</th><th>Canonical</th></tr></thead>
<tbody>{rh}</tbody></table></div></div></body></html>"""
    op=output_dir/"index.html"; op.write_text(h,encoding="utf-8"); return op

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--predictions",type=Path,default=None)
    ap.add_argument("--limit",type=int,default=None)
    args=ap.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    files=sorted(args.data.glob("*.json"))
    if args.limit: files=files[:args.limit]
    preds={}
    if args.predictions:
        with args.predictions.open() as f: preds=json.load(f)
    env=build_env(); rows=[]
    for p in files:
        try:
            t=load_trajectory(p); tid=t.get("trajectory_id",p.stem)
            op=render_trajectory(t,preds.get(tid),args.output,p,env)
            ce=t.get("canonical_expr") or "?"
            oe=t.get("original_expr") or "?"
            d=t.get("difficulty",0)
            reached=t.get("reached_canonical",len(t.get("steps",[]))==d)
            rows.append({"id":tid,"fn":op.name,"d":d,"v":"Success" if reached else "Failed","o":oe,"c":ce})
            print(f"Wrote: {op}")
        except Exception as e: print(f"Error {p}: {e}",file=sys.stderr)
    if rows: print(f"Wrote: {render_index(rows,args.output)}")

if __name__=="__main__": main()

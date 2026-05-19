# ISRE — Research Roadmap

## Baseline (уже есть)

| Policy | Success rate | Avg steps | Overhead vs optimal |
|--------|-------------|-----------|---------------------|
| Random | 99.2% | 6.69 | +3.62 |
| Greedy heuristic | 100% | 3.93 | +0.86 |
| BFS (depth ≤ 6) | TBD — надо измерить | TBD | TBD |
| **Neural target** | **≥ 98%** | **≤ greedy** | — |

Greedy — нижняя планка по interpretability. BFS — верхняя планка по оптимальности,
но только если он действительно решает ≥ 99% траекторий на нашем распределении.
**TODO:** прогнать BFS на 1K траекторий и зафиксировать реальный solve rate прежде
чем называть его "optimal ceiling".

---

## Методологические решения (явные)

### Параметрический бюджет: реальные числа от кода

Спека §2.6 называла «35–40M encoder, 8–10M policy» — это не совпадает с кодом.
Реальный подсчёт (encoder=TreeGRU 6 rounds + policy MLP/KAN):

| hidden_dim | encoder | policy_mlp | total |
|-----------|---------|------------|-------|
| 128 (prototype run) | 0.42M | 0.15M | **0.57M** |
| 256 | 1.65M | 0.56M | **2.21M** |
| 512 | 6.58M | 2.17M | **8.75M** |

Реальная прогрессия:
- Prototype: hidden=128 → 0.57M
- Research baseline: hidden=512 → 8.75M
- Scale-up: hidden=1024 → ~35M
- Upper bound: hidden=2048 → ~140M

Experiment 1 проводится при **hidden=512 (8.75M)**. Это «10M prototype» из спеки.
matched-interpretability: encoder одинаковый для MLP и KAN. Каждая архитектура
использует один и тот же 512-dim GRU output.

**Ablation: четыре строчки, не одна.** MLP-narrow и MLP-shallow отвечают на разные подвопросы:

| Вариант | Параметры policy head | Что тестирует |
|---------|----------------------|---------------|
| KAN (hand-crafted features) | ~0.07M | interpretable φ_i на ~25 scalar inputs |
| MLP-full (hidden=512) | 2.17M | верхняя планка MLP |
| MLP-narrow (hidden=128) | 0.15M | matched-params с KAN |
| MLP-shallow (hidden=512, 1 layer) | ~0.27M | inductive bias vs depth |

Без MLP-narrow и MLP-shallow critic выберет удобную интерпретацию.
С обоими — каждый подвопрос закрыт отдельной строчкой.

### Scope of evaluation

Success rate измеряется на распределении, где complexity metric монотонна —
expansion-heavy задачи исключены по конструкции генератора.
В репорте это должно быть explicit: "results hold within v1 evaluation distribution;
generalization to expansion-dominated tasks is out of scope for Experiment 1."

### Depth cap

TreeGRU rounds = max\_AST\_depth = 6. Это сознательное ограничение v1.
AMC/AIME-уровень (depth 8–10) — v2, требует либо динамических rounds,
либо другого encoder (e.g. Tree Transformer без fixed depth).
Зафиксировано в Open Questions ниже.

---

## Experiment 1

```
EXPERIMENT 1 — STATUS: FINDING ESTABLISHED

Primary claim: For this symbolic algebra task on v6_bfs distribution,
the bottleneck of greedy MLP policy is inference strategy, not model
capacity or learned representation quality.

Evidence (three independent axes, leak-free):
  - Capacity (FALSIFIED): hidden=512 <= hidden=128 across all 10 metrics
    on identical held-out split (split_seed=1234, trajectory-level)
  - Representation (FALSIFIED): gold-rank mean 1.23, median 1 across
    all rollout steps; model is locally near-optimal
  - Decoding (CONFIRMED): beam{2,3,5} reduces per-step error projection
    from 14.9% to {5.0%, 1.7%, 0.3%}; trajectory overhead reduced from
    0.129 to 0.011, BFS-optimal rate from 91.8% to 99.2%

Saturation note: success rate already at 100% under greedy; finding is
on path-optimality (efficiency), not success. This is the empirically
correct metric given the task structure.

Next experiments build on this finding:
  - Day-7 KAN synthetic feasibility check
  - Experiment 1 (KAN vs MLP) on hidden=128 with beam-{best} inference,
    KAN over hand-crafted features (not GRU embeddings)
  - 5-seed campaign on final configuration for mean±std
```

### Day-7 KAN synthetic feasibility (BLOCKING gate for KAN-vs-MLP)

Falsifiable check, not perfectionism: give KAN a task with KNOWN analytic
structure and verify φ_i curves recover it. Task: `score(x,y) =
sin(x) + |y|` — additive, two univariate ground-truth functions. PASS
iff φ_1 ≈ sin, φ_2 ≈ |·| (disentangled). If entangled → debug on the
synthetic, NOT on the real task. If KAN can't show interpretable φ_i on
a task with analytic ground truth, KAN-vs-MLP on the real task degrades
to "two models compared" — and at hidden=128 (already saturating) that
is uninteresting. Interpretability IS the reason KAN is in the project.

### KAN input = hand-crafted features, NOT GRU embeddings (architectural)

This is an MLP-vs-KAN-level architectural decision, not a nuance.
KAN φ_i curves are interpreted RELATIVE TO INPUT FEATURES. If KAN input
is the GRU encoder output, φ_i(embed_dim_17) is a black-box curve —
meaningless for interpretation. KAN input MUST be semantically
meaningful hand-crafted features (node_type, depth, subtree_size,
action_type, ...), so φ_depth(x) reads as "effect of node depth on
scoring". MLP keeps the GRU encoder; the comparison is then explicitly
"MLP-with-subtree-context vs KAN-with-readable-features".

### 5-seed campaign: ONE config, beam fixed (no greedy/beam split)

Fix beam-{best} (beam-2 or beam-3 by cost/benefit) as THE inference
setting for ALL seed runs BEFORE the campaign. mean±std is reported on
the final configuration only. Do NOT run 5×greedy and 5×beam separately
— that yields two groups with no defined "headline". One group, final
config.

**Датасет:** v6_bfs 48428 (gold=BFS-оптимум), trajectory-level split.
**Модели:** TreeGRU encoder (MLP) vs KAN over hand-crafted features.
**Метрики:** rollout success (saturated), step-overhead vs BFS-optimal
(primary), bfs_optimal_rate, MODE-B 3-way divergence, KAN φ_i curves.

### Rigor: seeds и variance

**5 независимых сидов** для каждой архитектуры. Не 3.

Причина не статистическая — политическая. KAN-вариативность по сидам это
известная претензия к оригинальной статье. Reviewer, который её знает, будет
смотреть именно на размах. С n=3 скажет "std не оценивается надёжно". С n=5 —
отстанет. Три сида — абсолютный floor, ниже не опускаться.

Репорт: mean ± std по success_rate и avg_steps.

На RTX 4070 Ti Super: 20 эпох × ~33 мин = ~11 часов на один run.
5 сидов × 2 архитектуры = ~110 часов GPU. Планировать заранее (~5 суток если
запускать последовательно; параллельно не выйдет на одной карте).

### Framing negative results

Если KAN φ\_i не интерпретируемы на задаче такого размера — это **публикуемое
наблюдение**, не провал. Все видели pykan на toy examples; на реальном символьном
reasoning никто не показывал. "KAN does not produce interpretable activations at this
scale" — это сильный результат, если подкреплён визуализацией и ablation.

Спека написана так, как будто результат будет положительным. Это неправильная
психология. Эксперимент должен быть честным независимо от исхода.

### Путь A — нейросеть бьёт greedy

Критерий: success rate ≥ 98%, avg steps < 3.93 на diff ≥ 4.

**Что дальше (масштабирование по осям):**

1. **Домен:** степени до 8, до 6 термов, коэффициенты до ±20.
2. **Датасет:** 500K траекторий, более равномерный difficulty (curriculum hardening).
3. **Анализ активаций:** KAN φ\_i для топ-3 действий — ищем монотонные /
   кусочно-полиномиальные паттерны. Это и есть главный исследовательский результат.
4. **Online RL:** поверх SL — REINFORCE или PPO с reward = –steps\_to\_canonical.
5. **Публикация:** если KAN-активации интерпретируемы и accuracy ≥ MLP → результат.

### Путь B — нейросеть не догоняет greedy

Не закрываем, а пробуем рычаги последовательно.

#### Рычаг 1: Архитектура
- Увеличить encoder (hidden dim 128→256).
- Попробовать другой KAN-бэкенд (pykan → efficient-kan → MonoKAN).
- Ensemble MLP+KAN с learnable mixing weight.
- **Стоп-критерий:** 3 варианта без роста → рычаг 2.

#### Рычаг 2: Усложнить домен
- Greedy слишком силён на простых полиномах: complexity монотонно падает,
  любой шаг улучшает. Степени 1–6 + 2 переменные создают локальные минимумы
  сложности, где жадный застревает.
- **Стоп-критерий:** greedy всё равно ≥ neural → рычаг 3.

#### Рычаг 3: Ограничить информацию жадного
- Greedy делает lookahead на 1 шаг. Restricted greedy — без lookahead (только тип
  действия по приоритетной таблице). Снижает success rate жадного, делает сравнение честнее.
- **Параллельно:** проверить интерпретируемость KAN независимо от accuracy.

#### Рычаг 4: Curriculum + длина траекторий
- max\_trajectory\_length до 10. Hard negatives где greedy ошибается.
- **Если ничего:** переформулировать как learning-to-rank, а не classification.

---

## Метрики для каждого эксперимента

Считать на rollout (не на train loss):

| Метрика | Описание |
|---------|----------|
| `success_rate` | Доля траекторий до канона за ≤ max\_steps |
| `avg_steps` | Среднее шагов при успехе |
| `step_overhead` | avg\_steps − avg\_optimal |
| `gold_rank@1` | Доля шагов где gold action = argmax политики |
| `invalid_rate` | Должно быть 0 (engine гарантирует валидность кандидатов) |
| `loop_rate` | Доля траекторий с повторяющимся состоянием |

KAN-specific: визуализировать activation curves φ\_i для топ-3 действий.

---

## Open Questions (v2+)

- **Depth scaling:** TreeGRU с fixed rounds не масштабируется на depth > 6.
  Варианты: динамические rounds (iterate until convergence), Tree Transformer,
  или рекуррентный encoder без depth limit. Нужно для AMC/AIME-уровня.
- **Multi-variable:** расширение на ℤ[x,y] требует нового canonical form и нового
  набора inverse transforms. Отдельный проект.
- **Online RL:** пока вся система supervised. RL поверх SL — следующий этап после
  того как baseline нейросеть сходится.

---

### KAN архитектурная развилка (решить до day 8)

Текущий KAN в policy.py: Linear(input→32) bottleneck + заглушка.
**Проблема:** GRU-энкодер уже перемешал фичи в 512-dim вектор. φ_i KAN будут
функциями от мутных комбинаций, не от структурных признаков. Interpretability убита до KAN.

| Вариант | Вход KAN | φ_i читаемы | Контекст поддерева |
|---------|----------|------------|-------------------|
| Текущий (bottleneck) | GRU(512)→Linear(32) | Нет | Да |
| **Hand-crafted features** | ~25 скаляров | **Да** | Нет |
| Grouped | GRU(64)+features(25) | Частично | Частично |

**Решение:** hand-crafted features для KAN (~25d: node_type_onehot, depth,
subtree_size, parent_type_onehot, action_type_onehot). MLP остаётся с GRU.
Сравнение честное: "MLP с контекстом поддерева vs KAN с читаемыми признаками".

**Day 7 mini-experiment (обязателен):** KAN на синтетических scalar входах
перед разворачиванием на реальных данных — страховка против "день 9 ничего нет".

---

## Текущий статус

Данные/движок (полная цепочка, см. POSTMORTEM):
- [x] Engine заморожен (`engine-frozen-v1.1`), 7 ENGINE_GAP-багов + критика-фиксы
- [x] Датасет v6 (`dataset-v6-validated`): v6_bfs 48428 (gold=BFS-оптимум,
      0% broken, 0% cycles, 10/10 действий) + v6_recorded (naive arm, спарен)
- [x] BFS baseline + bfs_optimal_path; baselines random 98.5% / greedy 100%
- [x] Per-index seeding (paired ablation); curriculum temperature-weighted
      (4-проверочная panel); global `--seed`; CAMPAIGN_SEEDS=[0,1,2,3,42]
- [x] Encoder векторизован (6.1x, eval-exact-equivalent, regression-тест)
- [x] **Trajectory-level leak-free split** (POSTMORTEM #7): val_traj_ids.json
      shared train↔eval, fail-loud, SPLIT_SEED=1234

Experiment 1 — primary finding УСТАНОВЛЕН (контролируемо, leak-free):
- [x] 128-full vs 512-full, единственная переменная hidden_dim →
      **capacity НЕ боттлнек** (512 хуже 128 по всем 10 метрикам)
- [x] gold-rank diagnostic → **representation НЕ боттлнек** (mean 1.23)
- [x] beam{1,2,3,5} eval (без ретрейна) → **decoding ЕСТЬ боттлнек**:
      overhead 0.129→0.011, bfs_optimal 91.8%→99.2% @ beam-5
- [x] eval_neural харнесс (MODE A rollout+beam / MODE B 3-way divergence)

Осталось (compute, ждёт go):
- [ ] (b) KAN-vs-MLP на **hidden=128** (точка насыщения) + beam-инференс —
      реальный Experiment 1; KAN с hand-crafted features (не GRU bottleneck)
- [ ] (d) 5-seed [1,2,3,42] с beam для mean±std на headline
- [ ] (c) опц. 256 — scaling-кривая 128/256/512
- [x] Day 7: KAN-синтетика **PASS** — f=sin(x)+|y|, R² φ_x=0.999, φ_y=0.997,
      additive-residual 0.024 (pykan 0.2.8). φ_i interpretability methodology VALID.
      Plot-bug RESOLVED: root cause was a 1-row forward poisoning pykan's cached
      postacts (std dof≤0→alpha nan), NOT a pykan defect. Fix = representative
      large-batch forward immediately before `.plot()` (pattern locked for (b)).
      Artifacts: phi_recovery.png (backend-independent) + pykan native sp_*.png.
- [~] KAN arm (b): smoke GREEN on 4 axes — matched-protocol EXACT (4843,
      split_seed=1234, == MLP@128), real KAN 7752p (not stub), val_acc
      0.768@e1 (≈ MLP@e6-8), φ smooth <0.5% roughness@e1. BLOCKER:
      7.24h/epoch (24x MLP) → plan infeasible as-is. NEXT: profile +
      vectorize KAN hot-path with equivalence-gate (encoder-precedent 6x),
      then reduced-epoch full + ladder {32,64,128}. NOT param-matched by
      design (KAN 7.7k vs MLP 475k: MLP=416k learned-encoder+58k head;
      KAN replaces encoder with 27 hand features — matched-PROTOCOL,
      capacity already controlled separately via 128≥512).
- [ ] Activation visualization для KAN φ_i (subgradient-equivalent fork-структура)

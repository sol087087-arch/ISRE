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

### Параметрический бюджет: matched-interpretability, не matched-params

Encoder одинаковый для MLP и KAN (~417K параметров).
Policy head: ~58K для обоих.
Это сравнение в режиме **matched-interpretability** — обе головы сжаты до одного
bottleneck, чтобы KAN мог оставаться читаемым.

Это *не* ответ на вопрос "что лучше при одинаковом бюджете в 100M параметров".
Это ответ на вопрос "даёт ли KAN-голова сопоставимую accuracy при сопоставимом
числе параметров, и при этом интерпретируемые активации?"

**Ablation: четыре строчки, не одна.** MLP-narrow и MLP-shallow отвечают на разные подвопросы:

| Вариант | Как matched | Что тестирует |
|---------|-------------|---------------|
| KAN | — | — |
| MLP-full | тот же hidden\_dim и depth что у основного MLP | верхняя планка MLP |
| MLP-narrow | уменьшен hidden\_dim до matched-params с KAN | KAN выигрывает за счёт expressiveness per param? |
| MLP-shallow | уменьшена глубина до matched-params с KAN | KAN выигрывает за счёт inductive bias на малой глубине? |

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

## Experiment 1: MLP vs KAN на v1-домене

**Датасет:** 50K траекторий, степени 1–4, 9 inverse-трансформов.
**Модели:** TreeGRU encoder (одинаковый) + MLP head vs KAN head.
**Метрики:** success rate на val rollout, avg steps, gold_rank@1.

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

## Текущий статус

- [x] Генератор 50K траекторий (все 9 inverse transforms, покрытие всех 9 action types)
- [x] Baselines: random, greedy
- [ ] BFS baseline + измерение реального solve rate
- [x] MLP train run: эпоха 1 — acc 85.9% train / 74.2% val (идёт на GPU)
- [ ] KAN head (реализовать, заменить заглушку в policy.py)
- [ ] 3-seed runs для MLP и KAN
- [ ] Evaluation/rollout script (success_rate на val)
- [ ] Activation visualization для KAN

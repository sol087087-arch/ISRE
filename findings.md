# ISRE — Research Findings Log

Хронологический лог находок по мере работы с пайплайном.
Каждая запись: дата, описание, последствия, статус.

---

## 2026-05-09 — Критические баги в symbolic engine (ENGINE_GAP)

### Контекст

При переходе на ground-truth сравнение в baseline-эвале (canonical_ast вместо
`len(get_candidates())==0`) success rate упал с 98.6% → 47.2% для random,
100% → 92.5% для greedy. Диагностика показала, что 48% random-провалов —
ENGINE_GAP: движок возвращал пустой список кандидатов для состояний, которые
НЕ являются канонической формой. Это не policy failure — это движок говорит
"не знаю что делать", хотя валидные трансформации существуют.

### Обнаруженные баги (все в `isre/symbolic/symbolic_engine.py`)

| # | Баг | Эффект |
|---|-----|--------|
| 1 | `_can_fold_const`: требовал ALL children = NUMBER. Не срабатывал на `Add(1, 2, x^4)` | FOLD_CONST gap |
| 2 | `_sort_key`: клал NUMBER (константы) ПЕРВЫМИ в ADD. Канон — константы ПОСЛЕДНИМИ. Движок считал `(-3 + (-7*x))` уже отсортированным | SORT_COMMUTATIVE = 0% в gold distribution |
| 3 | `_can_sort_commutative` / `_apply_sort_commutative`: один sort_key для и ADD, и MUL. В MUL NUMBER (коэффициент) должен идти ПЕРВЫМ, в ADD — ПОСЛЕДНИМ | MUL SORT gap |
| 4 | `_can_remove_zero`: только для ADD. `Mul(a, 0, b)` не редуцировался в 0 | MUL zero gap |
| 5 | `_can_remove_one`: не покрывал `Pow(x, 1) → x` | Pow trivial exponent gap |
| 6 | `_can_merge_power`: считал только POW-узлы, не VARIABLE. `Mul(x, x)` → не мержился | MERGE_POWER gap |
| 7 | `_can_fold_const` для MUL: требовал ALL = NUMBER. `Mul(3, 3, x)` → не фолдился | MUL FOLD gap |

### Эффект на baseline-числа

| Policy | До фикса | После фикса |
|--------|----------|-------------|
| Random success rate | 47.2% ± 2.7% | 97.6% |
| Greedy success rate | 92.5% | **100.0%** |
| Engine gap rate | 48% (random) / 8% (greedy) | **0.0%** |

### Эффект на тренировочные данные

SORT_COMMUTATIVE отсутствовал в gold action distribution (0% из 150K training pairs)
из-за бага #2: движок генерировал неправильный порядок сортировки, поэтому никогда
не создавал SORT_COMMUTATIVE как валидный gold action. Датасет 50K траекторий
обучает только 9 из 10 action types.

**Последствие:** MLP train run (эпохи 1-6, val_acc 74-86%) — результаты invalid.
Модель обучена на данных без SORT_COMMUTATIVE. Val_acc stagnation на 74% частично
объясняется curriculum expansion, частично — missing action type.

### Статус

- [x] Все 7 engine bagов исправлены
- [x] Diagnose_failures.py подтверждает ENGINE_GAP = 0%
- [ ] Датасет 50K нужно перегенерировать с исправленным engine
- [ ] MLP train run нужно перезапустить на новом датасете

### Для репорта

```
Initial baseline evaluation reported random=98.6%, greedy=100% success rate (n=500).
Investigation revealed _is_canonical was implemented as len(get_candidates())==0, which
conflated true canonical form with engine dead-end states. After switching to ground-truth
comparison against canonical_ast, numbers dropped to random=47.2%±2.7%, greedy=92.5%.

Failure diagnosis (diagnose_failures.py) revealed that 48% of random failures were
ENGINE_GAP (engine returned empty candidates for non-canonical states), not policy
failures. Seven bugs were identified in symbolic_engine.py, primarily: (a) SORT_COMMUTATIVE
sort order inconsistency between ADD and MUL nodes, (b) FOLD_CONST requiring all-NUMBER
children in mixed expressions, and (c) REMOVE_ZERO and MERGE_POWER not handling MUL edge
cases. After fixes: random=97.6%, greedy=100%, ENGINE_GAP=0.0%. The dataset (50K
trajectories) was generated before these fixes and contains 0% SORT_COMMUTATIVE training
examples; it must be regenerated before further training.
```

---

## Следующие шаги после этой находки

1. Перегенерировать датасет 50K с исправленным engine
2. Убедиться что SORT_COMMUTATIVE > 0% в gold action distribution
3. Запустить baseline comparison на новых данных (должен дать greedy ~100%, random ~97%)
4. Начать MLP train run заново

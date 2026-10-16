# Model card — term-deposit call ranking

**Author:** Ayazhan Aisan · **Course:** MOP 3231, Task 3 · **Seed:** 3042 · **scikit-learn** 1.7.2

## 1. Purpose
A bank can phone only **10 % of the clients on a list** in one campaign wave. The model gives every client a score (probability of saying *yes* to a term deposit) **before** any call is made, so that the call centre phones the 10 % with the highest scores.
It is a ranking tool for one marketing decision, not a judgement about any person, and not for credit or other decisions.

## 2. Data
- UCI *Bank Marketing* (`bank-additional-full.csv`, Moro, Rita & Cortez 2014, CC BY 4.0): Portuguese bank calls, May 2008 – Nov 2010. A teaching dataset.
- `train.csv`: 32 940 rows, 11.3 % *yes*. `test.csv`: 8 236 rows without the answer.
- **`duration` is not used.** It is the length of the call, known only after the call, so it would be leakage.
- No real NaN values. Missing answers are coded `"unknown"` (`default` 20.9 %, `education` 4.3 %, others < 3 %). `pdays = 999` means "never contacted before" (96.2 % of rows).

## 3. Pipeline (one scikit-learn `Pipeline`, fitted on training data only)
| Step | Columns | Treatment |
|---|---|---|
| Scaling | 8 numeric columns | `StandardScaler` |
| Sentinel | `pdays` | 999 → median of real values (6) + 0/1 indicator "never contacted", then scaled |
| Encoding | 10 categorical columns | `OneHotEncoder(handle_unknown="ignore")`, `"unknown"` kept as its own category |
| Model | 63 features | `RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, max_features=0.3, random_state=3042)` |

The saved file `model.joblib` takes the raw columns of `test.csv` (without `id`) and has `predict_proba`.

## 4. How it was chosen
- 20 % stratified hold-out kept aside; all comparison and tuning by 5-fold stratified CV on the other 80 %.
- Majority baseline: ROC-AUC 0.500, accuracy 0.887 (finds nobody). Logistic regression: ROC-AUC 0.794 ± 0.009.
- Tuned decision tree (0.791), KNN (0.791) and logistic regression (0.794) are within the noise of each other.
- Tuned random forest: **0.806 ± 0.006**, better than logistic regression in 5 of 5 folds (mean +0.012, std of the difference 0.0035).
- `class_weight="balanced"` changed ROC-AUC by < 0.002 (noise) and was not used.

## 5. Results on the untouched hold-out (6 588 clients, used once)
| Metric | Value |
|---|---|
| ROC-AUC | **0.790** |
| Average precision | **0.471** (random: 0.113) |
| Precision in the top 10 % | **0.508** (335 of 659 called say *yes*) |
| Recall in the top 10 % | **0.452** (335 of 742 buyers) |
| Lift over random calling | **4.5×** |

In plain words: about **one call in two** leads to a deposit, against about one in nine when calling at random.

## 6. Limits and ways it can fail in use
1. **Change of economic conditions.** The strongest signals are `nr.employed`, `euribor3m` and `emp.var.rate`. All data come from 2008–2010. If interest rates or employment move to values never seen in training, the scores can be badly wrong.
2. **Random split ≠ future.** Train and test cover the same period, so the 0.79 ROC-AUC is probably optimistic. A model trained on the past and used on a later campaign will most likely score lower.
3. **Changes in how the bank works.** New contact channels, a new product, different call months or a different list of clients change the patterns the model learned (e.g. the high yes-rate in March/Sept/Oct/Dec).
4. **Feedback loop.** Once only the top 10 % are called, the bank stops learning about the other 90 %; future training data will be biased towards clients the model already likes.
5. **New categories or missing columns.** A new job title is ignored (`handle_unknown="ignore"`), but a missing or renamed column makes the pipeline fail.
6. **Scores are rankings, not exact probabilities.** The 10 % cutoff must be recomputed on every new list.

## 7. What to monitor after launch
- **Precision of the called 10 %** in every wave (expected ≈ 0.5); alert if it falls toward the random rate (≈ 0.11).
- **Input drift:** distribution of `euribor3m`, `nr.employed`, `emp.var.rate`, `month`, `contact`, and the share of `"unknown"` values, compared with training.
- **Score distribution:** if the 90th-percentile cutoff moves a lot from 0.33, the client list or the economy has changed.
- **Retrain** on recent campaigns regularly, and keep a small random sample of clients called outside the top 10 % to measure what the model misses.S
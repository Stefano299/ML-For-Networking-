# Section 2 — Shared rules (3 people, one classifier each)

Only classifiers studied in the course so far (Lectures 5–9): linear models (Logistic Regression, SVM),
decision trees, neural networks. Random Forest / boosting are not covered yet: only as an extra.

| Who | Classifier | Notebook | Results files |
|---|---|---|---|
| A | PyTorch MLP (mandatory, attacked in Section 3–4) | `notebooks/02a_mlp.ipynb` | `mlp.csv` |
| B | Decision Tree (+ Random Forest as extra) | `notebooks/02b_tree.ipynb` | `decision_tree.csv` |
| C | Logistic Regression + SVM (linear on all data, RBF on a sample) + baselines + final comparison (2.3) | `notebooks/02c_linear_svm.ipynb` | `logistic_regression.csv`, `svm_linear.csv`, `svm_rbf.csv`, `baseline_*.csv` |

Everyone works **only on their own notebook**: never edit someone else's `.ipynb` (merge conflicts on notebooks are painful).

## Training, validation and test

- **Test set** (20%, from `load_train_test()`): used **once**, at the end, on the chosen classifier (point 2.3).
  Never for choosing hyperparameters.
- **Cross-validation** on the training set (80%), 5 folds: for each combination of hyperparameters the model is
  trained on 4 folds and scored on the 5th (the *validation fold*), 5 times. The mean of the 5 scores is the
  **validation score**.
- **Validation metric** = the metric used for that score: **F1 of class 1**. It decides the best
  hyperparameters of each model and which of the 3 classifiers is the best. Accuracy is not used for this: the
  app is from 3% to 34% of the flows, so a detector that always says 0 would already have 66–97% accuracy.

## What is fixed for everyone (do not change)

All in `src/data.py` and `src/evaluation.py`:

- **Data**: `load_train_test()` → same 80/20 split by `Flow.ID` (seed 42), 64 features, **not scaled**.
  It takes ~1 minute (it reads the 200 MB CSV), load it once at the top of the notebook.
- **Label**: `make_ovr_target(y_train, app)` → 0/1 for the app. All the 10 apps (`APPS`) are required.
- **Normalization inside each fold**: `with_scaler(model)` puts `LogMinMaxScaler` (log1p + Min-Max, the same as
  Section 1) before the model. In this way min and max are computed only on the training part of each fold and
  the validation fold stays really unseen. Hyperparameters are then called `model__<name>`.
- **CV folds**: `get_cv_folds(y_train, groups_train)` → 5 folds grouped by `Flow.ID` and stratified on the app.
  Pass them as `cv=folds`. Do not use `cv=5`: it would not group by `Flow.ID`.
- **Validation metric**: `SCORING = "f1"`.
- **Seed**: `SEED = 42` for `random_state` and `torch.manual_seed`.
- **Test metrics (2.3)**: `compute_metrics(y_true, y_pred, y_score)` → accuracy, precision, recall, F1,
  AUC-ROC (+ PR-AUC).
- **Saving**: `save_cv_result(model_name, app, search)` → one row per app in `results/section2/<model_name>.csv`.
  These CSVs are committed to git.

## Template — scikit-learn models (B, C)

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from src.data import APPS, load_train_test, make_ovr_target
from src.evaluation import SCORING, SEED, get_cv_folds, save_cv_result, with_scaler

X_train, X_test, y_train, y_test, groups_train = load_train_test()
folds = get_cv_folds(y_train, groups_train)

param_grid = {"model__max_depth": [5, 10, 20, None], "model__min_samples_leaf": [1, 10, 100]}
for app in APPS:
    y_bin = make_ovr_target(y_train, app)
    search = GridSearchCV(with_scaler(DecisionTreeClassifier(random_state=SEED)),
                          param_grid, scoring=SCORING, cv=folds, n_jobs=-1)
    search.fit(X_train, y_bin)
    save_cv_result("decision_tree", app, search)
```

Slow models (e.g. RBF SVM): tune on a sample of the training set taken with `group_sample` (whole flows,
stratified on the app), recompute the folds on the sample and write it in `notes`:

```python
from src.evaluation import group_sample

_, (X_s, y_s, g_s) = group_sample(X_train, y_train, groups_train, fraction=0.1)
folds_s = get_cv_folds(y_s, g_s)
...
save_cv_result("svm_rbf", app, search, notes="10% sample of the training set")
```

## Template — PyTorch MLP (A)

```python
import time
import numpy as np
from sklearn.metrics import f1_score
from src.data import LogMinMaxScaler, make_ovr_target
from src.evaluation import group_sample, save_cv_result

# X_train, y_train, groups_train, folds as in the template above; app = one of APPS, y_bin = make_ovr_target(y_train, app)
scores, fit_times, predict_times = [], [], []
for train_idx, val_idx in folds:
    # early-stopping set: 10% of the training part, whole flows (never the validation fold)
    (X_tr, yapp_tr, _), (X_es, yapp_es, _) = group_sample(
        X_train.iloc[train_idx], y_train.iloc[train_idx], groups_train.iloc[train_idx], fraction=0.1)
    scaler = LogMinMaxScaler().fit(X_tr)                            # fitted only on the training part
    X_tr, X_es = scaler.transform(X_tr), scaler.transform(X_es)
    X_val = scaler.transform(X_train.iloc[val_idx])
    y_tr = make_ovr_target(yapp_tr, app).to_numpy()
    y_es = make_ovr_target(yapp_es, app).to_numpy()
    y_val = y_bin.iloc[val_idx].to_numpy()

    t = time.time(); model = train_mlp(X_tr, y_tr, X_es, y_es, **params); fit_times.append(time.time() - t)
    t = time.time(); y_pred = predict(model, X_val); predict_times.append(time.time() - t)
    scores.append(f1_score(y_val, y_pred))
save_cv_result("mlp", app, best_params=params, cv_f1_mean=np.mean(scores), cv_f1_std=np.std(scores),
               fit_time=np.mean(fit_times), predict_time=np.mean(predict_times))
```

`train_mlp` / `predict` are yours (Adam, BCE loss, early stopping on `X_es`). The validation fold is never used
for early stopping, otherwise the validation F1 would be too optimistic compared to the other models.
Do the same split (`group_sample` on the whole `X_train`) for the final model.

## Checklist for each model (what Lectures 7–9 ask for)

Everyone, for their own model:

- [ ] Grid search with `cv=folds`, logarithmic ranges for parameters spanning orders of magnitude (e.g. `C`).
- [ ] Class imbalance as a hyperparameter: `class_weight` in `[None, "balanced"]` (MLP: `pos_weight` of
      `BCEWithLogitsLoss`). It matters for F1, and Section 3 requires a detector with a recall well above 0.
- [ ] `save_cv_result(...)` for **all 10 apps** (it also saves fit and predict time: efficiency, Lecture 8).
- [ ] **Validation curve** for the most important hyperparameter: `plot_validation_curve(...)`, and one sentence
      on underfitting/overfitting.
- [ ] If there are 2 main hyperparameters: **heatmap** `plot_grid_heatmap(search, x, y)` (e.g. `C`/`gamma` for
      the RBF SVM, `max_depth`/`min_samples_leaf` for the tree).
- [ ] Brief description of the best hyperparameters and a **rationale** for the results (required by 2.2).

Specific:

- **A (MLP)**: `plot_training_curve(train_loss, val_loss, best_epoch)` for the chosen configuration (early
  stopping, Lecture 9). Optimizer (e.g. Adam), learning rate, batch size and epochs are hyperparameters too.
- **B (tree)**: the feature importances, compared with the features found in the EDA.
- **C (linear + SVM)**: always include `class_weight="balanced"` in the grid. Measured on a 5% sample: without
  it, Logistic Regression and LinearSVC predict always 0 for the small apps (TWITTER: F1 0.00 → 0.12 with
  "balanced"; YOUTUBE: 0.60 → 0.67). `LinearSVC` has no `predict_proba`: use `decision_function` as `y_score`.
  `plot_learning_curve(...)` to justify the sample size used for the RBF SVM. Runs
  `run_baselines(X_train, y_train, folds, APPS)` once: majority class (F1 = 0, accuracy up to 97%) and random
  guess. Every model must be clearly above them.

Remember in the report: `cv_f1_std` is the variation between folds, not a confidence interval (Lecture 7).

## Final comparison (2.3) and handover

C runs `compare_models(load_all_cv_results())` when all the CSVs have the 10 apps (validation F1 per app and
mean over the 10 apps; the mean is NaN for an incomplete model). The best classifier is the one with the highest
mean, **decided only on validation**.

Then the owner of the best classifier, in their own notebook, retrains it on the whole `X_train` with the best
hyperparameters of each app (`GridSearchCV` already does it with `refit=True`; the MLP by hand) and evaluates it
**once** on the test set with `compute_metrics`, for the 10 apps. The test scores of the other models can be
reported too, but only as information: the choice is already fixed (Lecture 7: choosing on the test set turns
it into a validation set).

**A** delivers early a first version of the MLP for the group's application (weights, architecture config and
the `LogMinMaxScaler` fitted on the whole `X_train`, in `results/models/`): Sections 3 and 4 start from it.

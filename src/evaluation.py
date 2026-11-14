"""
Shared rules for Section 2, so the classifiers (MLP, Decision Tree, Logistic Regression / SVM) are compared
in the same way:
- same train/test split                  -> src.data.load_train_test()
- same normalization, fitted in each fold -> with_scaler() (Pipeline with LogMinMaxScaler)
- same cross-validation folds            -> get_cv_folds()
- same validation metric (F1 of class 1) -> SCORING
- same test metrics                      -> compute_metrics()
- same results file format               -> save_cv_result()
- baselines (majority / random guess)    -> run_baselines()
- plots for the report                   -> plot_validation_curve(), plot_grid_heatmap(),
                                            plot_learning_curve(), plot_training_curve()

Typical use with scikit-learn:

    X_train, X_test, y_train, y_test, groups_train = load_train_test()
    folds = get_cv_folds(y_train, groups_train)
    y_bin = make_ovr_target(y_train, app)
    search = GridSearchCV(with_scaler(DecisionTreeClassifier(random_state=SEED)),
                          {"model__max_depth": [5, 10, 20]}, scoring=SCORING, cv=folds)
    search.fit(X_train, y_bin)
    save_cv_result("decision_tree", app, search)

For the PyTorch MLP use the same `folds` in a manual loop, fitting LogMinMaxScaler on the training part of
each fold (see docs/project/SECTION2_RULES.md).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, learning_curve, validation_curve
from sklearn.pipeline import Pipeline

from src.data import LogMinMaxScaler, split_by_flow

N_FOLDS = 5
SEED = 42  # for the folds and for the models (random_state, torch.manual_seed)

# Validation metric = the score computed on the validation fold of the cross-validation, used to choose the
# hyperparameters and to compare the classifiers. Class 1 is always a minority (from ~3% to ~34% of the flows),
# so accuracy would reward a detector that always says 0. F1 of class 1 looks only at how well the app is
# detected (precision and recall together).
SCORING = "f1"

RESULTS_DIR = Path("results/section2")


def with_scaler(model):
    # Puts LogMinMaxScaler before the model: GridSearchCV fits it again in every fold, only on the training
    # part, so the validation fold is never used to compute min and max.
    # The hyperparameters of the model are then called "model__<name>" in the param grid.
    return Pipeline([("scaler", LogMinMaxScaler()), ("model", model)])


def get_cv_folds(y_train, groups_train, n_folds=N_FOLDS, seed=SEED):
    # Folds on the training set, built like the train/test split:
    # - grouped by Flow.ID, so rows of the same flow are never in both training and validation
    # - stratified on the app name (10 classes), so every app has about the same share in every fold
    #   and the same folds work for all the 10 detectors
    # Returns a list of (train_idx, val_idx) with positional indices, it can be passed as cv= to GridSearchCV.
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y_train)), y_train, groups_train))


def group_sample(X, y, groups, fraction, seed=SEED):
    # A fraction of the rows, taken by whole Flow.ID and stratified on the app (same logic as the train/test split).
    # Used for: (1) a sample of the training set for slow models (e.g. RBF SVM),
    # (2) the early-stopping set of the MLP inside a training fold.
    # Returns (rest, sample): two tuples (X, y, groups), with no Flow.ID in common.
    # The real fraction is 1 / round(1 / fraction) (e.g. 0.1 -> 10%, 0.15 -> 1/7 = 14%), because it is one fold.
    X_rest, X_s, y_rest, y_s, g_rest, g_s = split_by_flow(X, y, groups, test_size=fraction, seed=seed)
    return (X_rest, y_rest, g_rest), (X_s, y_s, g_s)


def compute_metrics(y_true, y_pred, y_score):
    # Test metrics required in 2.3. y_score is the score of class 1 (predict_proba[:, 1],
    # decision_function or the MLP sigmoid output), needed for the AUC-ROC.
    # We add the PR-AUC (average precision): with few positives it is more informative than the AUC-ROC.
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def save_cv_result(model_name, app, search=None, best_params=None, cv_f1_mean=None, cv_f1_std=None,
                   fit_time=np.nan, predict_time=np.nan, notes=""):
    # One CSV per model (results/section2/<model_name>.csv), one row per app.
    # Pass a fitted GridSearchCV/RandomizedSearchCV as `search`, or (for the MLP) the values by hand.
    # fit_time / predict_time: mean seconds per fold for the best hyperparameters (efficiency, Lecture 8).
    # With `search` they are taken from cv_results_ (predict_time = time to score the validation fold).
    # If the app is already there the row is replaced, so a cell can be re-run without duplicates.
    # Note: cv_f1_std describes the variation between folds, it is not a confidence interval
    # (the training parts of the folds overlap, Lecture 7).
    if search is not None:
        i = search.best_index_
        best_params = search.best_params_
        cv_f1_mean = search.cv_results_["mean_test_score"][i]
        cv_f1_std = search.cv_results_["std_test_score"][i]
        fit_time = search.cv_results_["mean_fit_time"][i]
        predict_time = search.cv_results_["mean_score_time"][i]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{model_name}.csv"
    row = pd.DataFrame([{
        "model": model_name,
        "app": app,
        "cv_f1_mean": float(cv_f1_mean),
        "cv_f1_std": float(cv_f1_std),
        "fit_time_s": float(fit_time),
        "predict_time_s": float(predict_time),
        "best_params": json.dumps(best_params, default=str),
        "notes": notes,
    }])
    if path.exists():
        old = pd.read_csv(path)
        row = pd.concat([old[old["app"] != app], row], ignore_index=True)
    row.to_csv(path, index=False)
    return row


def load_all_cv_results():
    # All the CSV of Section 2 together, for the final comparison (2.3)
    files = sorted(RESULTS_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No results in {RESULTS_DIR}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def compare_models(results):
    # Table app x model with the validation F1, plus the mean over the 10 apps.
    # The best classifier is the one with the highest mean (and we check it per app too).
    # The mean is NaN for a model that does not have all the apps yet, so incomplete models are not compared.
    table = results.pivot(index="app", columns="model", values="cv_f1_mean")
    table.loc["MEAN"] = table.mean(skipna=False)
    return table


def run_baselines(X_train, y_train, folds, apps):
    # Baseline hypotheses (Lecture 8: "always compare with a simple baseline"), same folds as the real models:
    # - majority: always predicts class 0 -> high accuracy but F1 = 0, it shows why accuracy is misleading
    # - stratified: random guess with the class proportions -> F1 ~ share of positives
    # A model is useful only if its validation F1 is clearly above these. Saved as baseline_*.csv.
    rows = []
    for strategy in ["most_frequent", "stratified"]:
        name = "baseline_majority" if strategy == "most_frequent" else "baseline_stratified"
        for app in apps:
            y_bin = (y_train == app).astype(int)
            cv = cross_validate(DummyClassifier(strategy=strategy, random_state=SEED), X_train, y_bin,
                                cv=folds, scoring=["f1", "accuracy"])
            acc = cv["test_accuracy"].mean()
            save_cv_result(name, app, best_params={}, cv_f1_mean=cv["test_f1"].mean(),
                           cv_f1_std=cv["test_f1"].std(), fit_time=cv["fit_time"].mean(),
                           predict_time=cv["score_time"].mean(), notes=f"cv_accuracy={acc:.4f}")
            rows.append({"model": name, "app": app, "cv_f1": cv["test_f1"].mean(), "cv_accuracy": acc})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# Plots for the report (Lecture 8). They all use the same folds and SCORING, and return the Axes.
# ---------------------------------------------------------------------------------------------------------------

def _param_label(v):
    # how a hyperparameter value is written on the plots (None and strings included)
    return f"{v:g}" if isinstance(v, (float, np.floating)) else str(v)


def plot_validation_curve(estimator, X, y, param_name, param_range, folds, log_scale=True, ax=None):
    # Training and validation F1 for different values of ONE hyperparameter (the others fixed):
    # both low -> underfitting, training high and validation low -> overfitting.
    # Non-numeric values (e.g. max_depth=None, class_weight="balanced") are shown as equally spaced categories.
    param_range = list(param_range)
    train_scores, val_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range,
                                                cv=folds, scoring=SCORING, n_jobs=-1)
    numeric = all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in param_range)
    x = param_range if numeric else np.arange(len(param_range))
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for scores, label in [(train_scores, "training"), (val_scores, "validation")]:
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(x, mean, marker="o", label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    if not numeric:
        ax.set_xticks(x, [_param_label(v) for v in param_range])
    elif log_scale:
        ax.set_xscale("log")
    ax.set_xlabel(param_name)
    ax.set_ylabel(f"{SCORING} (mean over folds)")
    ax.set_title("Validation curve")
    ax.legend()
    return ax


def plot_grid_heatmap(search, param_x, param_y, ax=None):
    # Validation F1 of a GridSearchCV with (at least) two hyperparameters as a heatmap, e.g. C and gamma of the
    # RBF SVM. If the grid has more parameters, for each (x, y) cell the best value over the others is shown.
    # Values are converted to text, otherwise pandas would silently drop None (e.g. max_depth=None);
    # the order is the one of the param grid.
    res = pd.DataFrame(search.cv_results_)
    xs = [_param_label(v) for v in res[f"param_{param_x}"]]
    ys = [_param_label(v) for v in res[f"param_{param_y}"]]
    table = (pd.DataFrame({"x": xs, "y": ys, "score": res["mean_test_score"]})
             .groupby(["y", "x"], sort=False)["score"].max().unstack("x"))
    table = table.reindex(index=list(dict.fromkeys(ys)), columns=list(dict.fromkeys(xs)))
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(table.to_numpy(dtype=float), cmap="viridis", origin="lower")
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    values = table.to_numpy(dtype=float)
    middle = (np.nanmin(values) + np.nanmax(values)) / 2
    for (r, c), v in np.ndenumerate(values):
        ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=8, color="black" if v > middle else "white")
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"Validation {SCORING} (grid search)")
    plt.colorbar(im, ax=ax)
    return ax


def plot_learning_curve(estimator, X, y, folds, train_sizes=(0.01, 0.05, 0.1, 0.3, 0.6, 1.0), ax=None):
    # Training and validation F1 when the model is trained on a growing fraction of each training fold
    # (the validation folds stay the same). If the validation curve is flat at the end, more data does not help:
    # useful to justify training slow models (e.g. RBF SVM) on a sample.
    sizes, train_scores, val_scores = learning_curve(estimator, X, y, train_sizes=list(train_sizes), cv=folds,
                                                     scoring=SCORING, shuffle=True, random_state=SEED, n_jobs=-1)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for scores, label in [(train_scores, "training"), (val_scores, "validation")]:
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, mean, marker="o", label=label)
        ax.fill_between(sizes, mean - std, mean + std, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("number of training flows")
    ax.set_ylabel(f"{SCORING} (mean over folds)")
    ax.set_title("Learning curve")
    ax.legend()
    return ax


def plot_training_curve(train_loss, val_loss, best_epoch=None, ax=None):
    # MLP: loss per epoch on the training data and on the early-stopping set (Lecture 9).
    # best_epoch = epoch kept by early stopping (vertical line).
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.arange(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="training")
    ax.plot(epochs, val_loss, label="early-stopping set")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", label=f"best epoch ({best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training curve")
    ax.legend()
    return ax

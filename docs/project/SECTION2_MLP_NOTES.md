# Section 2 — Role A: PyTorch MLP (work log)

> Author: Daniele Di Croce. Branch: `section2-mlp`. Files: `src/models.py`, `notebooks/02a_mlp.ipynb`,
> `results/section2/mlp.csv`, `results/section2/mlp_grid.csv`, `results/models/mlp_<app>.*`.
> This file documents what the MLP does and the design choices, so the group can review it and Sections 3–4
> can build on it. Updated as the work goes on.

## Status

| Step                                                                               | State                                  |
| ---------------------------------------------------------------------------------- | -------------------------------------- |
| `MLP` class, `train_mlp`, `predict` / `predict_proba`, `save_mlp` / `load_mlp`     | done                                   |
| First MLP for YOUTUBE saved in `results/models/` (early delivery for Sections 3–4) | done (v0), replaced by the final model |
| Cross-validation + grid search, `mlp.csv` for the 10 apps                          | done                                   |
| Validation curve, training curve, rationale                                        | done                                   |
| Final model for YOUTUBE in `results/models/`, informational test metrics           | done (see Results)                     |
| Speed-up of the CV: fold data computed once (done); bigger batches (optional)      | partly done                            |

## Open points for the group

1. **2 logits + `CrossEntropyLoss` instead of 1 logit + `BCEWithLogitsLoss`.** The template in `SECTION2_RULES.md`
   says "BCE loss, `pos_weight`"; `PROJECT_WORKFLOW_GUIDE.md` says "2 logits, `CrossEntropyLoss` with class weights,
   `nb_classes=2`". I followed the second, for these reasons:
   - **It is the same model.** With two output scores $z_0, z_1$, the softmax probability of class 1 is
     $\frac{e^{z_1}}{e^{z_0} + e^{z_1}} = \sigma(z_1 - z_0)$ (softmax: Lecture 10): the sigmoid of a single logit
     $h = z_1 - z_0$, i.e. the logistic regression of Lecture 06 / Lab 5. Same predictions, same metrics; only 33 redundant parameters
     more (6,306 vs 6,273 for `hidden=(64, 32)`).
   - **Same handling of the imbalance.** `pos_weight` of BCE multiplies the loss of the positives;
     `CrossEntropyLoss(weight=[w_0, w_1])` does the same with $w_0 = 1$, $w_1 =$ `pos_weight`. We use the
     scikit-learn "balanced" formula $w_c = m / (2 m_c)$, so `class_weight ∈ {None, "balanced"}` means the same
     thing for the MLP and for the sklearn models.
   - **It is what ART expects.** `PyTorchClassifier(nb_classes=2)` works with outputs of shape `(batch, nb_classes)`
     and one-hot labels with `nb_classes` columns; it is the format of ART's official `get_started_pytorch.py`
     cited by the project PDF. A single-output model is not the standard case and, depending on the ART version,
     needs label/loss adaptations or fails in the gradient computation of FGSM/PGD. Since the model is the same,
     the 2-logit form removes a risk in Section 3 at no cost.
   - The output is still raw logits, no softmax in `forward()` (`CODING_STYLE_GUIDE.md`).

   What changes for the rules: in the MLP template, "BCE loss" → "`CrossEntropyLoss`", "`pos_weight` of
   `BCEWithLogitsLoss`" → "`weight` of `CrossEntropyLoss`". If the group prefers the 1-logit form, it is a few lines
   in `train_mlp` / `predict_proba`.

## Decisions the group should know

Already taken and motivated, no answer needed; listed here because they touch the shared rules or Sections 3–4.

- **Group's app: free choice, currently YOUTUBE.** We derived YOUTUBE from the PDF rule (group number modulo 10,
  counting AD = 30, so app 0 of the list). On 2026-11-24 the professor said the choice of the app is free (see
  `tutor_clarifications_2026-11-24.md`), so it can still be changed. Changing it means rerunning the notebook with
  another `APP`: the grid on the new app and the final model.
- **`class_weight` of the YOUTUBE model: a tie, the unweighted model is saved.** At `lr=3e-3` the unweighted model
  has validation F1 0.818 ± 0.007 and the `balanced` one 0.817 ± 0.007. The rule (highest validation F1) picks the
  unweighted one: precision 0.84, recall 0.80 on the test set. The project PDF only asks that the detector "actually
  detects" the app, "e.g., a recall well above 0", so the requirement of Section 3 is met. The `balanced` variant
  has the same validation F1 with the opposite trade-off (precision 0.75, recall 0.91). To switch: one training of
  ~30 s, no cross-validation (see "Retrain from Python" with `class_weight="balanced"`). The choice must not be made
  by comparing the two test F1.
- **Scaler of the saved model: fitted on the training rows only.** `SECTION2_RULES.md` says in one place "the
  `LogMinMaxScaler` fitted on the whole `X_train`" and in another "do the same split (`group_sample` on the whole
  `X_train`) for the final model". We follow the second: the saved scaler is fitted on the 90% used for training
  (363,491 flows), not on the 10% early-stopping set, so the model is used with exactly the scaler it was trained
  with and the protocol is the same as inside the CV folds. With `clip=True` the outputs stay in [0, 1] in both
  cases. To switch: one line in 2.3 and one training.
- **The saved model is in git.** `.gitignore` has the exception `!results/models/mlp_*`, so
  `results/models/mlp_youtube.{pt,json}`, `mlp_youtube_scaler.joblib` and `mlp_youtube_params.json` (~80 KB in
  total) are committed and Sections 3–4 get them with a `git pull`.

## How it works (overview)

The detector is a small neural network that reads the 64 flow features (after the Section 1 preprocessing and
scaling) and outputs two scores, one for "other app" and one for "the app". The larger score is the prediction.

Training is the usual ERM: we look for the weights that minimize the average cross-entropy loss on the training
flows, using Adam (mini-batch gradient descent). To decide when to stop we watch the loss on a small held-out part of
the training data (early stopping): training stops when that loss stops improving, and we keep the weights of the best
epoch. Hyperparameters (layer sizes, learning rate, class weights) are chosen with the shared 5-fold cross-validation
of `SECTION2_RULES.md`, with F1 of class 1 as the metric.

The trained model is saved together with its scaler, so Sections 3–4 can reload it, feed it new flows scaled in the
same way, and wrap it in ART.

## Design (`src/models.py`)

### Architecture (class `MLP`)

```
input (64 features in [0, 1]) → Linear(64, h1) → ReLU → Linear(h1, h2) → ReLU → Linear(h2, 2) → 2 logits
```

- **Input**: the 64 features of `load_train_test()` (67 of Section 1 minus the 3 redundant ones), scaled by
  `LogMinMaxScaler` (log1p + Min-Max, fitted on the training part only).
- **Hidden layers** `hidden=(h1, h2)` is a hyperparameter (size of the model, Lecture 05/08). Each `Linear` is
  $\mathbf{w}^T\mathbf{x} + b$ as in the logistic regression of Lab 5; the `ReLU` ($\max(0, z)$) between layers makes the
  model non-linear. Without it, stacked linear layers would still be a linear classifier, and from the EDA a single
  feature separates YOUTUBE only with AUC ≈ 0.6, so a non-linear model is needed.
- **Output: 2 raw logits, no softmax in `forward()`**. `nn.CrossEntropyLoss` applies the (numerically stable)
  softmax itself, and ART computes input gradients from the logits (`CODING_STYLE_GUIDE.md`: "output raw logits").
  Class 0 = other apps, class 1 = the app.
- Parameters: 6,306 for `hidden=(64, 32)`, 16,706 for the chosen `hidden=(128, 64)` (vs 65 for a logistic
  regression on 64 features).
- **Why these sizes.** The usual starting point for tabular data: a first layer of the size of the input (64) or
  twice it (128), then a narrower one, so the network is forced to compress the 64 features towards the 2 logits;
  two hidden layers are enough (in theory one hidden layer can already approximate any continuous function, the
  universal approximation theorem; not covered in the lectures). With ~360k training flows
  there are 20–60 examples per weight, so neither size can memorize the training set (train F1 ≈ val F1). We did not
  try bigger or deeper nets: the step (64, 32) → (128, 64) gives only +0.01 F1, and a small model is faster to
  retrain many times in Sections 3–4.

### Training (`train_mlp`)

`train_mlp(X_tr, y_tr, X_es, y_es, hidden, lr, batch_size, max_epochs, patience, class_weight, seed, device)`
→ `(model, history)`. Inputs are numpy arrays already scaled (the caller fits `LogMinMaxScaler` on `X_tr`).

1. **Setup.** The seed is fixed (`torch.manual_seed(SEED)`, same `SEED = 42` as the other models), so the initial
   weights and the batch order are the same every run. The model is built with the requested `hidden` sizes and
   moved to the GPU if there is one (`get_device()`).
2. **Loss and optimizer.** Cross-entropy loss (the logistic loss of Lecture 06 for 2 classes), with optional
   class weights for the imbalance: `class_weight ∈ {None, "balanced"}`, same formula as scikit-learn
   ($w_c = m / (2 m_c)$), so errors on the minority class weigh more. It is a hyperparameter, as the rules checklist
   asks. The optimizer is Adam (Lecture 09) with learning rate `lr`.
3. **Epochs and batches** (Lecture 09, SGD). One epoch = one pass over the training rows, shuffled, in batches of
   `batch_size` rows. For each batch: forward pass + loss, backpropagation, one Adam step. With ~290k training rows
   per fold and `batch_size=512` that is ~570 updates per epoch.
4. **Early stopping** (Lecture 08/09). At the end of every epoch the loss is computed on the early-stopping set
   `X_es`. If it improved, a copy of the weights is kept; otherwise a counter grows, and after `patience` epochs
   without improvement training stops (or at `max_epochs`). The model returned has the weights of the best epoch.
   `X_es` is a 10% `group_sample` of the training part, **never the validation fold**, otherwise the validation F1
   would be optimistic compared to the other models.
5. **Output.** `history` has the training loss and the early-stopping loss per epoch and `best_epoch`: it is the
   input of `plot_training_curve` (`src/evaluation.py`).

### Inference and persistence

- `predict_proba(model, X)` → P(class 1) (softmax of the 2 logits, column 1), the `y_score` for AUC-ROC / PR-AUC.
- `predict(model, X, threshold=0.5)` → 0/1 labels; with 0.5 it equals the argmax of the logits, i.e. what ART's
  `PyTorchClassifier.predict()` does. Both run in batches with `torch.no_grad()`.
- `save_mlp(model, scaler, prefix)` writes `<prefix>.pt` (weights), `<prefix>.json` (architecture) and
  `<prefix>_scaler.joblib` (the scaler the model was trained with); `load_mlp(prefix)` rebuilds the same `MLP`,
  loads the weights and returns `(model, scaler)` (checked: identical predictions after a round-trip).

## Validation protocol (`02a_mlp.ipynb` 2.2)

- Same `folds` as everyone (`get_cv_folds`, grouped by `Flow.ID`, stratified on the app), manual CV loop
  (`cv_mlp`): in every fold a 10% `group_sample` of the training part is the early-stopping set, the scaler is fitted
  on the remaining 90%, the validation fold is only transformed and predicted. Metric: F1 of class 1. The std is the
  population std over the 5 folds (`ddof=0`), like `GridSearchCV`. The split and the scaling of each fold do not
  depend on the app or on the configuration, so they are computed once per fold and kept in memory (`get_fold_data`).
- **Grid on YOUTUBE**: `hidden ∈ {(64, 32), (128, 64)}` × `lr ∈ {1e-3, 1e-4}` × `class_weight ∈ {None, balanced}`
  (8 configs), then a **validation curve** on `lr ∈ {1e-4, 3e-4, 1e-3, 3e-3}` around the best one. Fixed: `batch_size=512`, `max_epochs=100`, `patience=5` (epochs chosen by early stopping).
- **Optimizer and batch size are fixed, not searched.** Adam adapts the step of every weight, so it needs less
  tuning than plain SGD, and its learning rate is searched. Batch size and learning rate act together (bigger batch,
  fewer updates per epoch, usually a larger lr), so the batch is fixed at a standard 512 (~570 updates per epoch)
  and only the lr is tuned. A bigger batch would also be faster, see "Where the CV time goes".
- **All 10 apps**: `hidden` of the YOUTUBE best; `lr ∈ {1e-3, 3e-3}` (the best of the grid and the best of the
  curve) × `class_weight ∈ {None, balanced}`, 4 configs per app, the best one kept. For YOUTUBE this adds the one
  combination the grid and the curve had not covered (`3e-3`, unweighted). Full tuning per app would cost ~10× more
  for models nobody attacks.
- The final YOUTUBE model uses the configuration written in `mlp.csv` (the best cached row of the app).
- Every CV result is appended to `results/section2/mlp_grid.csv` (one row per app × configuration, with train F1,
  mean best epoch and times, see "Cache" below); the best per app goes to `results/section2/mlp.csv` via
  `save_cv_result`, the format shared with the other classifiers.

### Hyperparameters searched

| Hyperparameter                 | Values tried                   | Searched on                                          |
| ------------------------------ | ------------------------------ | ---------------------------------------------------- |
| `hidden` (hidden layer sizes)  | (64, 32), (128, 64)            | YOUTUBE only; the other apps use (128, 64)           |
| `lr` (Adam learning rate)      | 1e-4, 3e-4, 1e-3, 3e-3         | all 4 on YOUTUBE; 1e-3 and 3e-3 on the other 9 apps   |
| `class_weight`                 | None, balanced                 | every app                                            |

Fixed for every configuration: optimizer Adam, `batch_size=512`, `max_epochs=100`, `patience=5`, `SEED=42`.

### Configurations run (5-fold CV each)

| Where                                    | Configurations                                                        | Count  |
| ---------------------------------------- | --------------------------------------------------------------------- | ------ |
| YOUTUBE, grid                            | `hidden` {(64, 32), (128, 64)} × `lr` {1e-4, 1e-3} × `class_weight`   | 8      |
| YOUTUBE, validation curve on `lr`        | (128, 64), balanced, `lr` 3e-4 and 3e-3 (1e-4 and 1e-3 from the grid) | 2      |
| YOUTUBE, search of the 10 apps           | (128, 64), `lr` 3e-3, unweighted                                      | 1      |
| each of the other 9 apps                 | (128, 64) × `lr` {1e-3, 3e-3} × `class_weight`                         | 4 × 9  |
| **Total** (rows of `mlp_grid.csv`)       |                                                                       | **47** |

YOUTUBE has 11 configurations: all 8 combinations of the grid plus 3 more with (128, 64). The grid heatmap in the
notebook shows the 8 grid configurations only (hidden × lr, best `class_weight` per cell); the other learning
rates are on the validation curve. The scores of the YOUTUBE configurations are in the table under "Results";
the scores of all 47 are in `mlp_grid.csv`.

## Results

### Before tuning (v0, `02a_mlp.ipynb` 2.1)

`hidden=(64, 32)`, `lr=1e-3`, `class_weight="balanced"`: F1 0.79 on the early-stopping set after 25 epochs.
It was the early delivery for Sections 3–4; the files in `results/models/` are now the final model below.

### Search on YOUTUBE: grid, lr curve, class weights (validation F1, mean ± std over 5 folds)

| hidden    | lr   | class_weight | val F1            | train F1 | epochs | s/fold |
| --------- | ---- | ------------ | ----------------- | -------- | ------ | ------ |
| (128, 64) | 3e-3 | None         | **0.818 ± 0.007** | 0.826    | 44     | 27     |
| (128, 64) | 3e-3 | balanced     | 0.817 ± 0.007     | 0.822    | 32     | 15     |
| (128, 64) | 1e-3 | balanced     | 0.811 ± 0.004     | 0.815    | 33     | 16     |
| (128, 64) | 1e-3 | None         | 0.810 ± 0.011     | 0.817    | 49     | 22     |
| (128, 64) | 3e-4 | balanced     | 0.799 ± 0.003     | 0.802    | 54     | 23     |
| (64, 32)  | 1e-3 | balanced     | 0.799 ± 0.006     | 0.801    | 40     | 19     |
| (64, 32)  | 1e-3 | None         | 0.791 ± 0.005     | 0.795    | 43     | 20     |
| (128, 64) | 1e-4 | None/bal.    | 0.790 ± 0.003–4   | 0.793–4  | 76–87  | 33–36  |
| (64, 32)  | 1e-4 | balanced     | 0.783 ± 0.003     | 0.785    | 98     | 40     |
| (64, 32)  | 1e-4 | None         | 0.773 ± 0.005     | 0.776    | 93     | 40     |

s/fold is the training time only; the `3e-3` unweighted row was measured in a later session with a busier GPU.
Plots: `results/plots/mlp_youtube_grid_heatmap.png` (the 8 grid configurations: hidden × lr, best `class_weight`),
`mlp_youtube_validation_curve_lr.png`, `mlp_youtube_training_curve_final.png`.

- **Learning rate matters most.** `1e-4` is too small: (64, 32) runs to the 100-epoch cap, (128, 64) stops after
  76–87 epochs, and both give the worst F1 at twice the cost. The validation curve is monotone up to `3e-3`, the largest
  value tried; the gain from `1e-3` to `3e-3` is within one std, so we did not go further. At `1e-4` training and
  validation F1 are both low and close: underfitting from a training that is too slow to converge, not from a model
  that is too small.
- **Bigger network helps a little** (+0.007 to +0.019 F1 from (64, 32) to (128, 64) at the same learning rate and class weight). We stopped there: the model is attacked
  in Sections 3–4 and smaller is simpler to reason about.
- **`balanced` makes no difference on YOUTUBE with the chosen network**: 0.818 unweighted vs 0.817 balanced at
  `3e-3`, 0.810 vs 0.811 at `1e-3`, all well inside one std. It helped only the small network (+0.01 for (64, 32)).
  The highest validation F1 is the unweighted row, so that is the final configuration (see "Decisions the group should know").
- **No overfitting:** training F1 is within 0.01 of validation F1 in every configuration, so early stopping on a
  held-out 10% does its job and more regularization (dropout, weight decay) was not needed.

### The other 9 apps (`results/section2/mlp.csv`)

`hidden=(128, 64)` for all; `lr` and `class_weight` are the best of the 4 cached rows per app. The two learning
rates give the same F1 within 0.035, with no consistent winner (6 of the other apps prefer `1e-3`, 3 prefer
`3e-3`): on the other apps the lr is not a decisive hyperparameter. Plot: `results/plots/mlp_validation_f1_per_app.png`. The
last column of the table is also in the notebook (table "Effect of the class weights" in 2.2, read from the cache).

| app            | class 1 share | lr   | class_weight | val F1        | with `balanced` (same lr) |
| -------------- | ------------- | ---- | ------------ | ------------- | ------------------------- |
| YOUTUBE        | 33.8%         | 3e-3 | None         | 0.818 ± 0.007 | 0.817                     |
| FACEBOOK       | 5.7%          | 1e-3 | None         | 0.868 ± 0.008 | 0.635                     |
| DROPBOX        | 5.0%          | 3e-3 | None         | 0.845 ± 0.019 | 0.634                     |
| AMAZON         | 17.2%         | 3e-3 | None         | 0.812 ± 0.008 | 0.776                     |
| WINDOWS_UPDATE | 6.8%          | 1e-3 | None         | 0.804 ± 0.014 | 0.674                     |
| CLOUDFLARE     | 2.9%          | 1e-3 | None         | 0.617 ± 0.035 | 0.428                     |
| GMAIL          | 8.0%          | 1e-3 | None         | 0.590 ± 0.008 | 0.459                     |
| MICROSOFT      | 10.8%         | 3e-3 | None         | 0.585 ± 0.007 | 0.517                     |
| SKYPE          | 6.1%          | 1e-3 | None         | 0.524 ± 0.007 | 0.372                     |
| TWITTER        | 3.6%          | 1e-3 | None         | 0.265 ± 0.030 | 0.200                     |

- **`balanced` never improves F1**: it is a tie on YOUTUBE and it hurts on every other app, sometimes a lot
  (FACEBOOK 0.87 → 0.64). With few positives the balanced weights push the model to predict the app much more often: recall goes up but precision collapses, and F1
  (the harmonic mean) drops. With F1 as the metric and a 0.5 threshold, unweighted training is the better default
  for rare classes; on YOUTUBE, where the two classes have comparable size, the weights are mild (1.48 and 0.76): they move the precision/recall trade-off but not the F1.
- **Difficulty follows Section 1.** The apps with generic, encrypted, short flows (TWITTER, SKYPE, MICROSOFT, GMAIL)
  are hard for every model; the ones with a distinctive traffic profile (FACEBOOK, DROPBOX, YOUTUBE, AMAZON,
  WINDOWS_UPDATE) reach F1 ≥ 0.80.

### Final model for YOUTUBE (`02a_mlp.ipynb` 2.3) and test set

`hidden=(128, 64)`, `lr=3e-3`, `class_weight=None`, `batch_size=512`, trained on the whole training set minus
a 10% `group_sample` for early stopping (363,491 + 40,388 flows): stopped after 44 epochs, best epoch 39, 20–30 s on GPU.
Saved as `results/models/mlp_youtube.{pt,json}`, `mlp_youtube_scaler.joblib` and `mlp_youtube_params.json` (the
hyperparameters, the best epoch and the CV score: 0.818 ± 0.007). Training curve:
`results/plots/mlp_youtube_training_curve_final.png`.

**Test set (informational only).** The rules use the test set once, on the classifier with the best _validation_ F1
among the three (chosen by role C with `compare_models`); these numbers do not enter any decision.

| metric    | test  |
| --------- | ----- |
| F1        | 0.815 |
| precision | 0.837 |
| recall    | 0.795 |
| accuracy  | 0.878 |
| AUC-ROC   | 0.954 |
| PR-AUC    | 0.919 |

Test F1 (0.815) is within one std of the CV estimate (0.818 ± 0.007), so the validation protocol was honest. The
detector makes the two kinds of error in similar numbers: it finds 80% of the YOUTUBE flows and 1 in 6 alarms is a false positive
(TN 61,523, FP 5,292, FN 7,001, TP 27,154). The previous `balanced` version of the final model, with the same
validation F1, had precision 0.753 and recall 0.914 (test F1 0.826): the class weights shift the decision towards
class 1, they do not make the model better. Confusion matrix: `results/plots/mlp_youtube_confusion_test.png`.

## Using and retraining the model

### Load and predict (Sections 3–4)

```python
from src.models import load_mlp, predict, predict_proba, train_mlp

model, scaler = load_mlp("results/models/mlp_youtube")   # model in eval mode, on GPU if available
X_test_scaled = scaler.transform(X_test)                  # same log1p + Min-Max as in training
y_pred = predict(model, X_test_scaled)                    # 0/1
y_score = predict_proba(model, X_test_scaled)             # P(class 1)
```

For ART (Section 3) wrap the same model:

```python
from art.estimators.classification import PyTorchClassifier
clf = PyTorchClassifier(model=model, loss=torch.nn.CrossEntropyLoss(), input_shape=(64,), nb_classes=2,
                        clip_values=(0.0, 1.0), device_type="gpu" if torch.cuda.is_available() else "cpu")
```

### Retrain from Python (adversarial training, poisoned training set of Section 4)

```python
import json
params = json.load(open("results/models/mlp_youtube_params.json"))   # hidden, lr, class_weight, batch_size, ...
(X_tr, yapp_tr, _), (X_es, yapp_es, _) = group_sample(X_train, y_train, groups_train, fraction=0.1)
scaler = LogMinMaxScaler().fit(X_tr)
model, history = train_mlp(scaler.transform(X_tr), make_ovr_target(yapp_tr, APP).to_numpy(),
                           scaler.transform(X_es), make_ovr_target(yapp_es, APP).to_numpy(),
                           hidden=tuple(params["hidden"]), lr=params["lr"], class_weight=params["class_weight"],
                           batch_size=params["batch_size"], max_epochs=params["max_epochs"], patience=params["patience"])
```

To train on a modified training set (poisoned flows, adversarial examples), pass that set as `X_tr` / `y_tr`
(`X_es` stays a clean `group_sample`). One training on the full training set takes 20–30 s on GPU, a few minutes on
CPU. No cross-validation is needed for this.

### Rerun the notebook

`notebooks/02a_mlp.ipynb` from the top, kernel: the project `.venv`. With the cache in place it takes ~6 minutes:
loading the data (~1 min), 2.1 and 2.3 each take a `group_sample` of the whole training set (~1.5–2 min each) and
train one model (20–30 s), 2.2 only reads the cache. 2.3 overwrites
`results/models/mlp_youtube.*`. The seed is fixed, so the result is the same up to GPU floating-point noise.
To do another app change `APP` in 2.0 (files take the app name, `mlp_<app>`; its grid costs ~1.5 min per
configuration, plus ~7 min for the fold data). Do not save the notebook from VS Code over a version executed from the terminal.

### Cache and when the CV must be redone

`run_config(app, params, folds)` looks in `results/section2/mlp_grid.csv` for a row with the same `app` and the
same `params` (JSON string); if found it returns it **without training**, otherwise it runs the 5-fold CV (~1.5–2 min)
and appends the row. The file is committed, so the group gets the results with a `git pull` and nobody needs a GPU.

- **To force a retraining**, delete the corresponding row(s) from `mlp_grid.csv` (or the whole file to redo
  everything: 47 rows, an estimated 1 h 15–1 h 30 on GPU now that the fold data is computed once; the two original
  runs took ~2 h 50 in total).
- The CV **must** be redone when the old numbers are no longer comparable: a change in the preprocessing or the
  feature set (`load_train_test`, `LogMinMaxScaler`, `REDUNDANT_FEATURES`), in the folds or `SEED`, or in the
  training procedure itself (loss, batch size, patience, 1-logit form).
- It must **not** be redone to retrain the final model, to use it with ART, or to change plots and text.

### Where the CV time goes, and how to make it faster

Measured on one fold (final configuration, RTX 5070):

| step                                                                    | time    |
| ----------------------------------------------------------------------- | ------- |
| `group_sample` of the training part (a 10-split `StratifiedGroupKFold`) | ~85 s   |
| fitting and applying `LogMinMaxScaler`                                  | ~2 s    |
| `train_mlp` (30–45 epochs)                                              | 15–25 s |
| predictions and F1                                                      | < 1 s   |

So the slow step was the early-stopping split, not the training, and it was repeated for every configuration
although it is the same for all of them (same fold, same seed): most of the ~2 h 50 of the first two runs went there.
**Done:** `get_fold_data` computes split and scaling once per fold and keeps the arrays in memory (~0.5 GB). The
computation is the same, so the cached rows stay valid. A new configuration now costs only its 5 trainings
(~1.5–2 min); the first one of a session also pays ~7 min for the 5 splits.

What is left is the training itself. The network is tiny, so each mini-batch of 512 rows keeps the GPU busy for
microseconds and the time goes into the fixed per-step overhead (~570 optimizer steps per epoch): the GPU sits at
~15% utilization, and a bigger GPU would not help, fewer steps would. Options, **not done**:

1. **`batch_size` 2048–4096** (one parameter in `FIXED`): 4–8× fewer steps per epoch, expected 4–6× faster.
   _Con:_ it changes the training (fewer updates per epoch), so it needs a larger `lr` (3e-3 or 1e-2), the F1 must
   be re-checked, and **the old cache is no longer comparable**: the grid has to be redone. It is a new experiment,
   not a free speed-up.
2. **One synchronization per epoch** in `train_mlp`: accumulate the batch losses as a GPU tensor and call `.item()`
   once per epoch. Identical results, ~20–30% faster, two lines.
3. **Drop `lr=1e-4`** from any new grid: worst F1 and slowest. No cost; the old rows document that it was tried.
4. Several trainings in parallel on the same GPU, or a lower `patience` / `max_epochs`: more code or a less fair
   comparison for a small gain. Discarded.

**Recommendation: nothing more now.** The cache is complete and the final model is saved; the CV is only redone when
something upstream changes (see the previous section). If that happens, apply 2 and 3; consider 1 only if the grid has
to grow, after a single-training check on YOUTUBE that `batch_size=4096` keeps the F1 within 0.005 of 512.

## Changelog

- 2026-11-15 — `MLP` class in `src/models.py`; this file.
- 2026-11-15 — `train_mlp`, `predict`, `predict_proba`, `save_mlp`, `load_mlp`; smoke test on synthetic data
  (GPU, early stopping, save/load round-trip, reproducibility with the seed).
- 2026-11-15 — `notebooks/02a_mlp.ipynb` 2.0–2.1: data, first YOUTUBE detector (v0) saved in `results/models/`,
  training curve in `results/plots/mlp_youtube_training_curve_v0.png`.
- 2026-11-16 — `02a_mlp.ipynb` 2.2–2.3: `cv_mlp` + cached `run_config` (`results/section2/mlp_grid.csv`), grid and
  lr curve on YOUTUBE, `class_weight` per app, `results/section2/mlp.csv` for the 10 apps, final YOUTUBE model with
  the best hyperparameters, informational test metrics. First run: ~104 min of CV on GPU.
- 2026-11-16 — fix: the final model takes the learning rate from the validation curve (`best_lr` = 3e-3), not from the
  heatmap only, so `mlp.csv`, `mlp_youtube_params.json` and the saved model agree. The other 9 apps were
  also evaluated with `(hidden, lr) = ((128, 64), 3e-3)` (18 configurations, ~65 min); with the `1e-3` rows, 46 rows
  in `mlp_grid.csv`, and `mlp.csv` keeps the best of the 4 per app.
- 2026-11-16 — review fixes. (1) `get_fold_data`: the early-stopping split and the scaling are computed once per fold
  instead of once per configuration (the split was most of the CV time). (2) The search is reproducible from an empty
  cache: the heatmap and the "best of the grid" use only the 8 grid rows, and `lr ∈ {1e-3, 3e-3}` × `class_weight` is
  run explicitly for all 10 apps. (3) That adds YOUTUBE `(128, 64)`, `3e-3`, unweighted: 0.818 ± 0.007, a tie with
  `balanced` (0.817) and the new best row, so the final model is now unweighted (test F1 0.815, precision 0.837,
  recall 0.795). (4) Corrected in this file: lr preference of the other apps (6 vs 3), the `1e-4` epochs, the claim
  that `balanced` helps on YOUTUBE, the scaler comment in `save_mlp`. 47 rows in `mlp_grid.csv`.
- 2026-11-16 — `02a_mlp.ipynb` 2.2: table of the validation F1 without / with `balanced` for the 10 apps at their
  best learning rate (from the cache). Notebook rerun from the top with the cache: same `mlp.csv`, same final model
  (best epoch 39, test F1 0.815).
- 2026-11-16 — notes: "Open points" keeps only what waits for an answer (app, 2 logits + `CrossEntropyLoss`); the
  class-weight tie, the scaler and the `.gitignore` exception moved to "Decisions the group should know".
- 2026-11-17 — `02a_mlp.ipynb` 2.2 and these notes: why optimizer and batch size are fixed, and the
  underfitting at small learning rates on the validation curve (Section 2 checklist). Text only, no rerun.
- 2026-11-19 — notes: tables of the hyperparameters searched and of the 47 configurations run.
- 2026-11-24 — notes: the group's app is no longer an open point (the professor said the choice is free);
  lecture references checked against the lecture notes (universal approximation is not in the lectures).

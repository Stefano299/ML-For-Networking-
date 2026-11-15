# Section 2 — Role A: PyTorch MLP (work log)

> Author: Daniele Di Croce. Branch: `section2-mlp`. Files: `src/models.py`, `notebooks/02a_mlp.ipynb`,
> `results/section2/mlp.csv`, `results/models/mlp_<app>.*`.
> This file documents what the MLP does and the design choices, so the group can review it and Sections 3–4
> can build on it. Updated as the work goes on.

## Status

| Step                                                                                        | State                            |
| ------------------------------------------------------------------------------------------- | -------------------------------- |
| `MLP` class (architecture)                                                                  | done                             |
| `train_mlp` (Adam, early stopping, class weights)                                           | done                             |
| `predict` / `predict_proba`                                                                 | done                             |
| `save_mlp` / `load_mlp`                                                                     | done                             |
| First MLP for the group's app, saved in `results/models/` (early delivery for Sections 3–4) | done (v0, fixed hyperparameters) |
| Cross-validation + grid search, `mlp.csv` for the 10 apps                                   | to do                            |
| Validation curve, training curve, rationale                                                 | to do                            |

## Open points for the group

1. **Group's app.** Group AD: counting letters like spreadsheet columns, AD = 30, 30 mod 10 = 0 → **YOUTUBE**.
   To be confirmed with the tutor (Franco Galante) before Section 3 starts.
2. **2 logits + `CrossEntropyLoss` instead of 1 logit + `BCEWithLogitsLoss`.** The template in `SECTION2_RULES.md`
   says "BCE loss, `pos_weight`"; `PROJECT_WORKFLOW_GUIDE.md` says "2 logits, `CrossEntropyLoss` with class weights,
   `nb_classes=2`". I followed the second, for these reasons:
   - **It is the same model.** With two output scores $z_0, z_1$, the softmax probability of class 1 is
     $\frac{e^{z_1}}{e^{z_0} + e^{z_1}} = \sigma(z_1 - z_0)$: the sigmoid of a single logit $h = z_1 - z_0$, i.e.
     the logistic regression of Lecture 06 / Lab 5. Same predictions, same metrics; only 33 redundant parameters
     more (6,306 vs 6,273).
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
3. ~~`.gitignore` excluded the saved model~~ — solved: `.gitignore` now has the exception `!results/models/mlp_*`, so
   `results/models/mlp_youtube.{pt,json}` and `mlp_youtube_scaler.joblib` (~35 KB in total) are committed and
   Sections 3–4 get them with a `git pull`.

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

## Design

### Architecture (`src/models.py`, class `MLP`)

```
input (64 features in [0, 1]) → Linear(64, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 2) → 2 logits
```

- **Input**: the 64 features of `load_train_test()` (67 of Section 1 minus the 3 redundant ones), scaled by
  `LogMinMaxScaler` (log1p + Min-Max, fitted on the training part only).
- **Hidden layers** `hidden=(64, 32)` is a hyperparameter (size of the model, Lecture 05/08). Each `Linear` is
  $\mathbf{w}^T\mathbf{x} + b$ as in the logistic regression of Lab 5; the `ReLU` ($\max(0, z)$) between layers makes the
  model non-linear. Without it, stacked linear layers would still be a linear classifier, and from the EDA a single
  feature separates YOUTUBE only with AUC ≈ 0.6, so a non-linear model is needed.
- **Output: 2 raw logits, no softmax in `forward()`**. `nn.CrossEntropyLoss` applies the (numerically stable)
  softmax itself, and ART computes input gradients from the logits (`CODING_STYLE_GUIDE.md`: "output raw logits").
  Class 0 = other apps, class 1 = the app.
- Parameters: 6,306 weights for `hidden=(64, 32)` (vs 65 for a logistic regression on 64 features).

### Training (`train_mlp`)

`train_mlp(X_tr, y_tr, X_es, y_es, hidden, lr, batch_size, max_epochs, patience, class_weight, seed, device)`
→ `(model, history)`. Inputs are numpy arrays already scaled (the caller fits `LogMinMaxScaler` on `X_tr`).

What happens inside, in order:

1. **Setup.** The seed is fixed (`torch.manual_seed(SEED)`, same `SEED = 42` as the other models), so the initial
   weights and the batch order are the same every run (checked: two runs give identical predictions). The model is
   built with the requested `hidden` sizes and moved to the GPU if there is one (`get_device()`).
2. **Loss and optimizer.** Cross-entropy loss (the logistic loss of Lecture 06 for 2 classes), with optional
   class weights for the imbalance: `class_weight ∈ {None, "balanced"}`, same formula as scikit-learn
   ($w_c = m / (2 m_c)$), so errors on the minority class (the app) weigh more. It is a hyperparameter, as the rules
   checklist asks. The optimizer is Adam (Lecture 09) with learning rate `lr`.
3. **Epochs and batches** (Lecture 09, SGD). One epoch = one pass over the training rows, shuffled, in batches of
   `batch_size` rows. For each batch: forward pass + loss, backpropagation of the gradient, one Adam step on the
   weights. With ~320k training rows and `batch_size=512` that is ~630 updates per epoch.
4. **Early stopping** (Lecture 08/09). At the end of every epoch the loss is computed on the early-stopping set
   `X_es`. If it improved, a copy of the weights is kept; otherwise a counter grows, and after `patience` epochs
   without improvement training stops (or at `max_epochs`). The model returned has the weights of the best epoch.
   `X_es` is a 10% `group_sample` of the training part of the fold, **never the validation fold**, otherwise the
   validation F1 would be optimistic compared to the other models.
5. **Output.** `history` has the training loss and the early-stopping loss per epoch and `best_epoch`: it is the
   input of `plot_training_curve` (`src/evaluation.py`).

### Inference and persistence

- `predict_proba(model, X)` → P(class 1) (softmax of the 2 logits, column 1), the `y_score` for AUC-ROC / PR-AUC.
- `predict(model, X, threshold=0.5)` → 0/1 labels; with 0.5 it equals the argmax of the logits, i.e. what ART's
  `PyTorchClassifier.predict()` does. Both run in batches with `torch.no_grad()`.
- `save_mlp(model, scaler, prefix)` writes `<prefix>.pt` (weights), `<prefix>.json` (architecture) and
  `<prefix>_scaler.joblib` (the `LogMinMaxScaler` fitted on the whole `X_train`); `load_mlp(prefix)` rebuilds the
  same `MLP`, loads the weights and returns `(model, scaler)` (checked: identical predictions after a round-trip).

### First detector for YOUTUBE (v0, `02a_mlp.ipynb` 2.1)

Fixed hyperparameters, not tuned: `hidden=(64, 32)`, `lr=1e-3`, `batch_size=512`, `class_weight="balanced"`,
early stopping on a 10% `group_sample` of the training set (patience 5). Stopped after 25 epochs (best epoch 20),
16 s on GPU. On the early-stopping set (optimistic, it chose the number of epochs): F1 0.79, recall 0.87,
precision 0.72, AUC-ROC 0.94, with 33.8% of positives. Enough to start Sections 3–4; the real validation score comes
from the cross-validation below. Saved as `results/models/mlp_youtube.{pt,json}` + `mlp_youtube_scaler.joblib`
(the scaler fitted on the 90% used for training, so model and scaler are consistent).

### Validation (planned, `02a_mlp.ipynb`)

- Same `folds` as everyone (`get_cv_folds`, grouped by `Flow.ID`, stratified on the app), manual CV loop from
  `SECTION2_RULES.md`, metric F1 of class 1, results via `save_cv_result("mlp", app, ...)`.
- Grid (small, training is on GPU but 10 apps × 5 folds × configs adds up):
  `hidden ∈ {(64, 32), (128, 64)}`, `lr ∈ {1e-3, 1e-4}`, `class_weight ∈ {None, balanced}`; batch size and max
  epochs fixed (epochs chosen by early stopping).

## How to use the model (for Sections 3–4)

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

To retrain (adversarial training by hand, backdoor in Section 4): `train_mlp(scaler.transform(X_tr), y_tr,
scaler.transform(X_es), y_es, **best_params)`, with `X_es` a `group_sample` of the training set.

## How to retrain

- **Same model again**: run `notebooks/02a_mlp.ipynb` from the top (kernel: the project `.venv`). Section 2.1 takes
  ~1 minute to load the data and ~20 s to train on GPU (a few minutes on CPU), and overwrites the files in
  `results/models/`. The seed is fixed, so the result is the same.
- **Other hyperparameters / another app**: change `params` or `APP` in 2.1; the saved files take the app name
  (`mlp_<app>`). Everything else stays the same.
- **From Python** (Sections 3–4, e.g. adversarial training or the poisoned training set of Section 4):

  ```python
  (X_tr, yapp_tr, _), (X_es, yapp_es, _) = group_sample(X_train, y_train, groups_train, fraction=0.1)
  scaler = LogMinMaxScaler().fit(X_tr)
  model, history = train_mlp(scaler.transform(X_tr), make_ovr_target(yapp_tr, APP).to_numpy(),
                             scaler.transform(X_es), make_ovr_target(yapp_es, APP).to_numpy(), **params)
  ```

  To retrain on a modified training set (poisoned flows, adversarial examples), pass that set as `X_tr`/`y_tr`.

## Changelog

- 2026-11-15 — `MLP` class in `src/models.py`; this file.
- 2026-11-15 — `train_mlp`, `predict`, `predict_proba`, `save_mlp`, `load_mlp`; smoke test on synthetic data
  (GPU, early stopping, save/load round-trip, reproducibility with the seed).
- 2026-11-15 — `notebooks/02a_mlp.ipynb` sections 2.0–2.1: data, first YOUTUBE detector (v0) saved in
  `results/models/`, training curve in `results/plots/mlp_youtube_training_curve_v0.png`.

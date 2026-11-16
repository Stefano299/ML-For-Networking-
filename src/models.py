"""
PyTorch MLP for Section 2 (role A), reused in Sections 3-4 (ART attacks, adversarial training, backdoor).

The network takes the 64 features scaled in [0, 1] by LogMinMaxScaler (src/data.py) and returns 2 raw
logits (class 0 = other apps, class 1 = the app). No softmax in forward(): nn.CrossEntropyLoss and ART's
PyTorchClassifier both expect unnormalized logits.

Typical use:

    scaler = LogMinMaxScaler().fit(X_tr)
    model, history = train_mlp(scaler.transform(X_tr), y_tr, scaler.transform(X_es), y_es, lr=1e-3)
    y_pred = predict(model, scaler.transform(X_val))
    save_mlp(model, scaler, "results/models/mlp_youtube")
    model, scaler = load_mlp("results/models/mlp_youtube")
"""
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from src.evaluation import SEED


def get_device():
    # GPU if available, otherwise CPU. The same code runs on both.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    # Multi-layer perceptron: input -> hidden layers with ReLU -> 2 logits.
    # hidden = sizes of the hidden layers, e.g. (64, 32) gives 64 -> 64 -> 32 -> 2.
    # It is the logistic regression of Lab 5 (a linear map w^T x + b) with non-linear layers in between,
    # so the decision boundary is no longer a hyperplane.

    def __init__(self, n_features, hidden=(64, 32)):
        super().__init__()
        self.n_features = n_features
        self.hidden = tuple(hidden)
        layers = []
        n_in = n_features
        for n_out in self.hidden:
            layers.append(nn.Linear(n_in, n_out))  # w^T x + b, n_out neurons
            layers.append(nn.ReLU())               # max(0, z): the non-linearity
            n_in = n_out
        layers.append(nn.Linear(n_in, 2))          # output layer: one logit per class
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: tensor (batch, n_features) -> logits (batch, 2)
        return self.net(x)


def _to_tensor(X, device):
    return torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)


def class_weights(y, class_weight):
    # Weights of the two classes for the loss, same formula as scikit-learn's class_weight="balanced":
    # w_c = m / (2 * m_c), so the minority class counts more and the two classes weigh the same in total.
    # class_weight=None -> both 1 (plain cross-entropy).
    if class_weight is None:
        return torch.ones(2)
    if class_weight != "balanced":
        raise ValueError("class_weight must be None or 'balanced'")
    counts = np.bincount(np.asarray(y), minlength=2)
    return torch.as_tensor(len(y) / (2 * counts), dtype=torch.float32)


def train_mlp(X_tr, y_tr, X_es, y_es, hidden=(64, 32), lr=1e-3, batch_size=512, max_epochs=100,
              patience=5, class_weight=None, seed=SEED, device=None):
    """
    ERM with the cross-entropy loss, solved with mini-batch gradient descent (Adam) and early stopping.

    X_tr, y_tr: training part, already scaled in [0, 1] (LogMinMaxScaler fitted on X_tr only)
    X_es, y_es: early-stopping set, scaled with the same scaler. It decides when to stop and which epoch
                to keep; it is never the validation fold of the cross-validation
    hidden, lr, batch_size, class_weight: hyperparameters (model size, step size, batch, imbalance)
    max_epochs, patience: training stops after `patience` epochs without improvement of the loss on
                the early-stopping set, or at max_epochs

    Returns the model with the weights of the best epoch (on the device) and a history dict with the
    training loss and the early-stopping loss per epoch, and best_epoch (1-based), for plot_training_curve.
    """
    device = device or get_device()
    torch.manual_seed(seed)  # same initial weights and same batch order every time
    generator = torch.Generator(device=device).manual_seed(seed)

    X_tr, X_es = _to_tensor(X_tr, device), _to_tensor(X_es, device)
    y_tr = torch.as_tensor(np.asarray(y_tr), dtype=torch.long, device=device)
    y_es = torch.as_tensor(np.asarray(y_es), dtype=torch.long, device=device)

    model = MLP(X_tr.shape[1], hidden).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_tr.cpu().numpy(), class_weight).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    m = len(X_tr)
    history = {"train_loss": [], "es_loss": [], "best_epoch": 0}
    best_loss, best_state, epochs_without_improvement = np.inf, None, 0

    for epoch in range(1, max_epochs + 1):
        # --- one epoch = one pass over the training set in random mini-batches (SGD, Lecture 9) ---
        model.train()
        order = torch.randperm(m, generator=generator, device=device)
        epoch_loss = 0.0
        for start in range(0, m, batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()                     # forget the gradients of the previous batch
            loss = criterion(model(X_tr[idx]), y_tr[idx])  # forward: average loss on the batch
            loss.backward()                           # backpropagation: gradient of the loss w.r.t. all weights
            optimizer.step()                          # w <- w - lr * (Adam-corrected) gradient
            epoch_loss += loss.item() * len(idx)
        history["train_loss"].append(epoch_loss / m)

        # --- loss on the early-stopping set, with the weights at the end of the epoch ---
        model.eval()
        with torch.no_grad():
            es_loss = criterion(model(X_es), y_es).item()
        history["es_loss"].append(es_loss)

        # --- early stopping: keep the best weights, stop when they don't improve for `patience` epochs ---
        if es_loss < best_loss:
            best_loss, best_state = es_loss, copy.deepcopy(model.state_dict())
            history["best_epoch"], epochs_without_improvement = epoch, 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_proba(model, X, batch_size=65536):
    # P(class 1) for each row of X (scaled), as a numpy array: the y_score for AUC-ROC / PR-AUC.
    # softmax turns the 2 logits into probabilities that sum to 1; column 1 is the app.
    device = next(model.parameters()).device
    model.eval()
    X = _to_tensor(X, device)
    probs = []
    with torch.no_grad():  # inference only: no gradients, less memory
        for start in range(0, len(X), batch_size):
            logits = model(X[start:start + batch_size])
            probs.append(torch.softmax(logits, dim=1)[:, 1])
    return torch.cat(probs).cpu().numpy()


def predict(model, X, threshold=0.5):
    # Predicted label (0/1): class 1 if P(class 1) >= threshold. With 0.5 it is the argmax of the logits,
    # i.e. what ART's PyTorchClassifier.predict() does.
    return (predict_proba(model, X) >= threshold).astype(int)


def save_mlp(model, scaler, prefix):
    # Saves everything needed to reuse the detector in Sections 3-4, with the given prefix:
    #   <prefix>.pt             weights (state_dict)
    #   <prefix>.json           architecture (n_features, hidden), to rebuild the same MLP
    #   <prefix>_scaler.joblib  the LogMinMaxScaler the model was trained with (fitted on the rows used for training)
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), prefix.with_suffix(".pt"))
    with open(prefix.with_suffix(".json"), "w") as f:
        json.dump({"n_features": model.n_features, "hidden": list(model.hidden)}, f)
    joblib.dump(scaler, f"{prefix}_scaler.joblib")


def load_mlp(prefix, device=None):
    # Loads the model (in eval mode, on the device) and the scaler saved by save_mlp.
    # Use: X_scaled = scaler.transform(X); y_pred = predict(model, X_scaled)
    device = device or get_device()
    prefix = Path(prefix)
    with open(prefix.with_suffix(".json")) as f:
        config = json.load(f)
    model = MLP(config["n_features"], config["hidden"]).to(device)
    model.load_state_dict(torch.load(prefix.with_suffix(".pt"), map_location=device))
    model.eval()
    scaler = joblib.load(f"{prefix}_scaler.joblib")
    return model, scaler

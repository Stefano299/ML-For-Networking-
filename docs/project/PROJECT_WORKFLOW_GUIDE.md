# ML4N Project — Operational & Workflow Development Guide

> **Target Audience:** AI Assistants (Claude, Cursor, Copilot) & Students working on the ML4N Group Project.  
> **Course:** Machine Learning for Networking (Politecnico di Torino — Prof. Luca Vassio).  
> **Repository Context:** Tabular IP Flow Classification, Adversarial Robustness, and Backdoor Security.

---

## 1. Project Architecture & Codebase Mapping

The repository is structured to separate reproducible library code (`src/`), exploratory experiments (`notebooks/`), and documentation (`docs/`):

```text
ML Lab/
├── data/
│   ├── raw/dataset.csv                     # Original Kaggle dataset
│   └── processed/
│       ├── dataset_10apps.csv              # Filtered dataset (10 applications)
│       └── class_distribution_10apps.csv   # Class balance statistics
├── src/                                    # Reusable modular logic
│   ├── data.py                             # Filtering, stratified sampling, scaling, splits
│   ├── models.py                           # PyTorch MLP architecture and training routines
│   ├── attacks.py                          # Noise, ART wrapper, FGSM, PGD, constraints, backdoor
│   └── evaluation.py                       # Metrics tables, robust accuracy, evasion rates, ASR
├── notebooks/                              # Phase-by-phase execution and reporting
│   ├── 01_eda.ipynb                        # EDA, correlation heatmaps, feature inspection
│   ├── 02_models.ipynb                     # Decision Tree, RF/SVM, PyTorch MLP training & CV
│   ├── 03_adversarial.ipynb                # Random noise, FGSM, PGD, constrained attacks, adv training
│   └── 04_backdoor.ipynb                   # Backdoor poisoning and unsupervised clustering defense
└── docs/                                   # Authoritative knowledge base
    ├── README.md                           # Master navigation index for AI
    ├── CODING_STYLE_GUIDE.md               # Professor coding standards and conventions
    ├── project/                            # Specifications and workflow guide
    ├── theory/                             # Lecture theory notes (Lectures 2, 3, 4)
    └── labs/                               # Reference lab code (Lab 2, 3, 4)
```

---

## 2. Theory-to-Project Bridging Guide

### A. Feature Types & Preprocessing (Course Lecture 3)
- **Categorical / Nominal:** Protocol (TCP=6, UDP=17), TCP Flags $\to$ Use `pd.get_dummies` or `OneHotEncoder`. Drop redundant baseline column if using linear models to avoid multicollinearity.
- **Continuous / Ratio:** Packet lengths, inter-arrival times, window sizes, byte counts $\to$ Non-negative, heavy-tailed.
- **Min-Max Scaling $[0, 1]$ is Mandatory:**
  $$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
  *Why?* The adversarial perturbation budget $\varepsilon$ is defined under the Euclidean $L_2$ norm:
  $$\|\mathbf{x} - \mathbf{x}_{\text{adv}}\|_2 \le \varepsilon$$
  Without $[0, 1]$ normalization, a perturbation of $\varepsilon=0.1$ on a port number (scale $0-65535$) would be negligible, while on a probability feature it would be catastrophic.

### B. Exploratory Data Analysis & Outliers (Course Lecture 2)
- **Boxplot & IQR rule:** Detect outliers with $1.5 \times \text{IQR}$. In network flows, outliers often represent large file transfers (elephants) vs frequent short control flows (mice).
- **Pearson vs. Spearman Correlation:**
  - Pearson ($r$) detects strictly linear relationships.
  - Spearman ($r_s$) evaluates monotonic relationships using ranks.
  - If pairs of features have $|r| > 0.95$ (e.g., total packets and total bytes in uniform streams), one is redundant.

### C. Dimensionality Reduction & Visualization (Course Lecture 4)
- **PCA for Global Spread:** Visualizing the high-dimensional flow dataset in 2D/3D via `PCA(n_components=2)`. Verify whether the positive application separates linearly from the background.
- **t-SNE Warning:** Use t-SNE *only* for non-linear 2D visualization. **Do NOT** use t-SNE features as input for classifiers (t-SNE is non-parametric, non-invertible, and does not preserve global distances).

---

## 3. Step-by-Step Implementation Workflow

### Step 1: Data Preparation (`src/data.py` & `notebooks/01_eda.ipynb`)
1. Filter the 10 target applications:
   `["YOUTUBE", "AMAZON", "GMAIL", "WINDOWS_UPDATE", "MICROSOFT", "FACEBOOK", "DROPBOX", "SKYPE", "TWITTER", "CLOUDFLARE"]`.
2. Extract the binary target for your group's application ($\text{Group ID} \pmod{10}$):
   - Positive ($y=1$): target app.
   - Negative ($y=0$): other 9 apps.
3. Compute the class distribution to document class imbalance (typically positive class is $5\%-15\%$).
4. Partition data into Train/Validation/Test sets using **Stratified Sampling** (`StratifiedShuffleSplit` or `train_test_split(..., stratify=y)`).
5. Fit `MinMaxScaler` on Train set only; transform Validation and Test sets.

### Step 2: Classifier Benchmarking (`src/models.py` & `notebooks/02_models.ipynb`)
1. Train 3 baseline models:
   - **Decision Tree** (`DecisionTreeClassifier(max_depth=...)`): Interpretable benchmark.
   - **Random Forest / SVM** (`RandomForestClassifier` or `SVC`): Strong tabular ensemble.
   - **PyTorch MLP (`nn.Module`)**: Multi-layer perceptron (e.g., Input $\to$ 64 $\to$ ReLU $\to$ 32 $\to$ ReLU $\to$ 2 logits).
2. Train PyTorch MLP using `nn.CrossEntropyLoss` with class weights if imbalance is severe:
   ```python
   weights = torch.tensor([1.0, (y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)
   criterion = nn.CrossEntropyLoss(weight=weights)
   ```
3. Evaluate models in `src/evaluation.py` reporting Precision, Recall (Class 1), F1-Score, and ROC-AUC.

### Step 3: Adversarial Robustness Evaluation (`src/attacks.py` & `notebooks/03_adversarial.ipynb`)
1. **Random Noise Baseline:** Add uniform random vectors scaled to $\|\mathbf{\delta}\|_2 = \varepsilon$, clipped to $[0, 1]$.
2. **ART PyTorchClassifier:**
   ```python
   from art.estimators.classification import PyTorchClassifier

   art_classifier = PyTorchClassifier(
       model=model,
       loss=criterion,
       optimizer=optimizer,
       input_shape=(n_features,),
       nb_classes=2,
       clip_values=(0.0, 1.0)
   )
   ```
3. **Attacks with ART:**
   - **FGSM:** `FastGradientMethod(estimator=art_classifier, eps=epsilon, norm=2)`.
   - **PGD:** `ProjectedGradientDescent(estimator=art_classifier, eps=epsilon, norm=2, max_iter=20)`.
4. **Constrained Realistic Attacks:**
   - Define boolean mask for controllable features (e.g., packet padding bytes $\ge 0$, flow delay $\ge 0$).
   - Clip perturbations to prevent illegal physical modifications (e.g., reducing transmitted byte counts).
5. **Adversarial Training:** Retrain using `AdversarialTrainer(art_classifier, pgd_attack)`.

### Step 4: Backdoor Injection & Defender Clustering (`src/attacks.py` & `notebooks/04_backdoor.ipynb`)
1. **Backdoor Attack:**
   - Select a controllable feature (e.g., `Init_Win_bytes_forward`).
   - For a percentage of Class 1 training samples ($0.1\%$ to $5\%$), inject a trigger value (e.g., fixed maximum window) and set label to $y=0$.
   - Train poisoned model; test Attack Success Rate (ASR) on triggered Class 1 test samples.
2. **Defender Clustering Defense:**
   - Without knowing the trigger, cluster training flows using $K$-Means and DBSCAN/GMM.
   - Compute cluster purity and identify outlier clusters containing anomalous distributions.
   - Drop suspect clusters, retrain the model, and verify recovery of clean accuracy and mitigation of ASR.

---

## 4. Key Pitfalls & Best Practices

| Pitfall | Problem | Solution |
| :--- | :--- | :--- |
| **Data Leakage in Scaling** | Fitting `MinMaxScaler` on the full dataset before split | Always `fit()` scaler on Train only, then `transform()` Validation & Test |
| **Scikit-Learn MLP for Attacks** | Attempting to run ART FGSM/PGD on `sklearn.neural_network.MLPClassifier` | ART requires PyTorch / TensorFlow to compute input gradients $\nabla_{\mathbf{x}} \mathcal{L}$. Always use PyTorch MLP |
| **Softmax in Model Output** | Applying `nn.Softmax` inside PyTorch MLP `forward()` | Output raw logits; `nn.CrossEntropyLoss` and ART expect unnormalized logits |
| **Ignoring Imbalance** | Relying on Accuracy when Class 1 is $8\%$ of the dataset | Use ROC-AUC, F1-Macro, and explicit Class 1 Recall as primary metrics |
| **Unrealistic Perturbations** | Perturbing IP addresses or negative packet bytes | Use ART `mask` parameter and non-negative delta constraints |

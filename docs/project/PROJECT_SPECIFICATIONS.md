# ML4N — Group Project 4: Analyzing Adversarial Attacks on Network Data Classifiers

> **Course:** Machine Learning for Networking (ML4N) — Politecnico di Torino  
> **Academic Year:** 2024/2025  
> **Course Instructor:** Prof. Luca Vassio  
> **Project Tutor / Referent:** Franco Galante (`franco.galante@polito.it`)  
> **Domain:** Tabular Machine Learning Security & Robustness on IP Network Traffic Flows  

---

## 1. Scenario & Problem Statement
Machine Learning classifiers play a vital role in modern network security and traffic management (e.g., detecting applications based on packet sizes, inter-arrival times, flag counts, and flow durations). 

However, classifiers trained on tabular network data are vulnerable to adversarial perturbations. An attacker seeks to evade detection by crafting an adversarial flow sample $\mathbf{x}_{\text{adv}}$ from a clean flow $\mathbf{x}$ such that:
$$\|\mathbf{x} - \mathbf{x}_{\text{adv}}\|_2 \le \varepsilon$$
where $\varepsilon$ denotes the maximum allowable perturbation budget (Euclidean distance $L_2$) to ensure the attack remains realistic, stealthy, and within protocol validity.

---

## 2. Section 1 — Data Acquisition, Preprocessing & Exploration

### 1. Data Acquisition & Filtering
- **Dataset:** *IP Network Traffic Flows Labeled with 75/87 Apps* (Kaggle).
- **Application Filtering:** Filter the dataset to include strictly the **10 target applications**:
  ```python
  APPS = [
      "YOUTUBE", "AMAZON", "GMAIL", "WINDOWS_UPDATE", "MICROSOFT",
      "FACEBOOK", "DROPBOX", "SKYPE", "TWITTER", "CLOUDFLARE"
  ]
  ```
- **Stratified Random Sampling:** If your computational resources are limited, apply stratified sampling to downsample the dataset to a computationally manageable size while strictly maintaining the original class frequency proportions.

### 2. Data Preprocessing & One-vs-Rest Setup
- **Feature Sanitization:** Remove uninformative identifier columns (e.g., `Flow ID`, exact IP addresses, raw timestamps) that would cause data leakage or overfitting.
- **Categorical Encoding:** Encode categorical features using One-Hot Encoding.
- **Min-Max Normalization:** Normalize all numerical features into the range $[0, 1]$:
  $$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
  *(Fit the scaler strictly on the training set, then transform the test set).*
- **One-vs-Rest Binary Target:**
  For each group's assigned application (determined by $\text{Group ID} \pmod{10}$):
  - **Class 1 (Positive):** The assigned target application.
  - **Class 0 (Negative):** All other 9 network applications.
- **Class Imbalance Analysis:** Calculate and report the percentage share of positive samples for each of the 10 classes.

### 3. Exploratory Data Analysis (EDA)
- Analyze feature distributions via **Histograms / KDE** and **Boxplots**.
- Compute correlation matrices (**Pearson** linear correlation and **Spearman** rank correlation) and visualize with heatmaps.
- Identify the most discriminative network flow features across classes.

---

## 3. Section 2 — Supervised Data Analysis & Model Training

### 1. Classifier Selection
- Train at least **three distinct classifiers** covered in the course (e.g., Decision Trees, Random Forest / SVM, Multi-Layer Perceptron).
- **MANDATORY REQUIREMENT:** At least one model must be a **Neural Network (MLP) implemented in PyTorch**.
  > *Technical Note:* Scikit-learn's `MLPClassifier` does not support gradient backpropagation for external attack libraries. Adversarial Robustness Toolbox (ART) requires a PyTorch model wrapped via `art.estimators.classification.PyTorchClassifier` to perform gradient-based attacks (FGSM, PGD).

### 2. Validation & Hyperparameter Tuning
- Employ **Stratified $K$-Fold Cross-Validation** to respect class imbalance across folds.
- Select evaluation metrics suitable for imbalanced datasets (do not rely on Accuracy alone).
- Document tuning rationales and selected hyperparameter sets.

### 3. Model Evaluation on Independent Test Set
- Evaluate clean performance on the unseen test set.
- Report full classification metrics:
  - **Accuracy**
  - **Precision**
  - **Recall** (with particular focus on **Class 1 Recall**)
  - **F1-Score** (Macro and Weighted)
  - **ROC-AUC Score**
- Display Confusion Matrices.

---

## 4. Section 3 — Adversarial Attacks & Robustness Evaluation

> **Assigned Application:** Group Number $\pmod{10}$. The PyTorch detector must be optimized for high initial recall on Class 1.

### 1. Random Noise Baseline
- Generate uniform/Gaussian random noise perturbations constrained by Euclidean norm budget:
  $$\|\mathbf{\delta}\|_2 = \varepsilon \quad \implies \quad \mathbf{x}_{\text{adv}} = \text{clip}(\mathbf{x} + \mathbf{\delta}, 0, 1)$$
- Evaluate **Robust Accuracy** across the entire test set.
- Measure **Class 1 Evasion Rate** (percentage of Class 1 flows misclassified as Class 0).
- **Feature Sensitivity Analysis:** Perturb features individually to determine feature vulnerability and find the minimum $\varepsilon$ required to evade detection.

### 2. Gradient-Based Attacks with ART
- Wrap the PyTorch model with `art.estimators.classification.PyTorchClassifier(clip_values=(0.0, 1.0))`.
- Execute and compare:
  - **FGSM (Fast Gradient Sign Method):** Single-step attack under $L_2$ norm.
  - **PGD (Projected Gradient Descent):** Iterative multi-step attack under $L_2$ norm.
- Evaluate attack efficacy across a range of $\varepsilon$ values.
- Identify which features undergo the largest perturbations.

### 3. Constrained / Domain-Realistic Attacks
- **Physical Network Traffic Constraints:**
  In real-world network packets:
  - Byte count cannot be reduced ($\Delta \ge 0$, packet padding only).
  - Packet count cannot be reduced ($\Delta \ge 0$).
  - Inter-arrival times cannot be reduced without physical speedup ($\Delta \ge 0$, packets can only be delayed).
  - Immutable headers (e.g., protocol types, fixed flags) cannot be altered arbitrarily.
- **Implementation:**
  - Utilize ART's `mask` argument to restrict perturbations to controllable features.
  - Clip perturbations so changes only move in physically valid directions.
- Compare robust accuracy, evasion rates, and minimum $\varepsilon$ between unconstrained and domain-constrained attacks.

### 4. Countermeasures: Adversarial Training
- Retrain the detector using adversarial training via `art.defences.trainer.AdversarialTrainer`.
- Re-evaluate clean accuracy, robust accuracy, and Class 1 detection rate against clean and adversarial flows.

---

## 5. Section 4 — Attacks on Unsupervised Data Analysis (Backdoor & Clustering)

### 1. Backdoor Poisoning Attack
- **Trigger Injection:**
  - Inject a specific trigger pattern into a controllable feature (e.g., a specific rare watermark in `Init_Win_bytes_forward` or packet length).
  - Inject the trigger into a fraction of Class 1 flows during training, but flip their ground-truth labels to **Class 0**.
- Retrain the detector across varying poisoning fractions (e.g., $0.1\%, 0.5\%, 1.0\%, 5.0\%$).
- **Evaluation:**
  - **Clean Test Accuracy:** Verify model performance remains high on unpoisoned traffic.
  - **Attack Success Rate (ASR):** Measure the percentage of triggered Class 1 test flows misclassified as Class 0.

### 2. Defender Clustering & Anomaly Isolation
- **Scenario:** The defender suspects training set poisoning but does not know the exact trigger or feature.
- **Clustering Techniques:** Apply at least **two unsupervised clustering methods** (e.g., $K$-Means, DBSCAN, or Gaussian Mixture Models - GMM) on the training set.
- **Defense Evaluation:**
  - Measure cluster purity and identify outlier/suspicious clusters containing poisoned samples.
  - Compute the fraction of poisoned flows successfully pruned vs. clean flows false-positively dropped.
  - Retrain the classifier on the sanitized training set and verify that the backdoor vulnerability is removed while maintaining high clean accuracy.

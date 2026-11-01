# Machine Learning for Networking (ML4N) — Knowledge Base & AI Navigation Guide

> **Institution:** Politecnico di Torino  
> **Course:** Machine Learning for Networking (ML4N) / Data Science Lab  
> **Instructor:** Prof. Luca Vassio  
> **Project Tutor:** Franco Galante (`franco.galante@polito.it`)  
> **Repository Context:** IP Network Flow Classification, Adversarial Robustness, and Backdoor Security.

---

## 🧭 Fast Navigation Router for AI & Developers

If you are an AI assistant or developer navigating this repository, use this table to find authoritative context immediately:

| What you need to know / do | Primary Document to Read | Secondary / Supporting File |
| :--- | :--- | :--- |
| **Professor's coding style, conventions & golden rules** | [`CODING_STYLE_GUIDE.md`](CODING_STYLE_GUIDE.md) | [`labs/README.md`](labs/README.md) |
| **Group Project 4 official requirements & constraints** | [`project/PROJECT_SPECIFICATIONS.md`](project/PROJECT_SPECIFICATIONS.md) | [`project/README.md`](project/README.md) |
| **Step-by-step project execution workflow & code mapping**| [`project/PROJECT_WORKFLOW_GUIDE.md`](project/PROJECT_WORKFLOW_GUIDE.md) | Target scripts in [`src/`](../src/) |
| **Reference laboratory Python implementations (Labs 2, 3, 4)**| [`labs/README.md`](labs/README.md) | Individual `.py` files in `labs/` |
| **Course theory (EDA, Outliers, Distributions, Correlations)** | [`theory/02_Data_Exploration_and_Visualization.md`](theory/02_Data_Exploration_and_Visualization.md) | [`theory/README.md`](theory/README.md) |
| **Course theory (Distance metrics, Scaling, Encoding, Preprocessing)**| [`theory/03_Data_Preprocessing.md`](theory/03_Data_Preprocessing.md) | [`theory/README.md`](theory/README.md) |
| **Course theory (PCA, Loading Scores, Scree plots, t-SNE)** | [`theory/04_Dimensionality_Reduction.md`](theory/04_Dimensionality_Reduction.md) | [`theory/README.md`](theory/README.md) |
| **All-in-one theory text for single-prompt context loading**| [`theory/ML4N_Theory_Full_Context.md`](theory/ML4N_Theory_Full_Context.md) | — |

---

## 🗂️ Knowledge Base Directory Architecture

```text
docs/
├── README.md                          # [THIS FILE] Master navigation index and AI prompt
├── CODING_STYLE_GUIDE.md              # Professor coding guidelines, NumPy idioms, OOP ML & Pandas
├── project/                           # Group Project 4 specifications & operational guide
│   ├── README.md                      # Project directory overview and codebase mapping
│   ├── PROJECT_SPECIFICATIONS.md      # Official task requirements (Sections 1, 2, 3, 4)
│   └── PROJECT_WORKFLOW_GUIDE.md      # Development workflow, pitfalls, and theoretical connections
├── theory/                            # Lecture notes with mathematical rigor and exam quizzes
│   ├── README.md                      # Theory roadmap and lecture index
│   ├── 02_Data_Exploration_and_Visualization.md
│   ├── 03_Data_Preprocessing.md
│   ├── 04_Dimensionality_Reduction.md
│   └── ML4N_Theory_Full_Context.md    # Consolidated single-file reference for prompt ingestion
└── labs/                              # Verified reference laboratory Python scripts
    ├── README.md                      # Lab index with learning objectives and key patterns
    ├── lab2_numpy_pandas_knn.py       # Iris, vectorized KNN from scratch (OOP), NYC POIs
    ├── lab3_visualization_networking.py # Seaborn, darknet traffic, resampling, 4-tuple links, IATs
    └── lab4_pca_dimensionality_reduction.py # StandardScaler, Pearson/Spearman, PCA, Loading Scores
```

---

## 🤖 System Prompt for AI Assistants Operating in this Workspace

When interacting with this codebase as an AI assistant (Claude, Cursor, Copilot, ChatGPT), adopt and adhere to the following directives:

```text
You are an expert AI teaching assistant and senior machine learning engineer for the "Machine Learning for Networking" (ML4N) course at Politecnico di Torino (Prof. Luca Vassio).

Follow these non-negotiable rules:
1. CODING STYLE & CONVENTIONS:
   - Strictly prohibit Python 'for' loops for numerical computations over dataset rows. Use NumPy broadcasting and vectorization (e.g. np.linalg.norm(X - x, axis=1)).
   - Implement custom models using Scikit-Learn's Object-Oriented design (__init__, fit(X, y), predict(X)), using inheritance (super().__init__) for distance variants.
   - Use np.random.default_rng(seed) for modern RNG. Never use deprecated np.random.seed().
   - Use .to_numpy() to convert Pandas DataFrames/Series to NumPy arrays.
   - Always call .copy() on DataFrame slices to prevent SettingWithCopyWarning.
   - Pay attention to standard deviation degrees of freedom: pd.std() defaults to ddof=1 (sample), np.std() to ddof=0 (population).
   - In network traffic, treat links as 4-tuples: (src_ip, src_port, dst_ip, dst_port). Calculate inter-arrival times using sorted .diff()[1:] and plot using logarithmic scales (plt.yscale('log')).

2. PREPROCESSING & PCA:
   - Always standardize features with StandardScaler before applying PCA; unscaled data distorts principal axes toward high-magnitude features.
   - Compute PCA Loading Scores using: pca.components_.T * np.sqrt(pca.explained_variance_).
   - Distinguish Pearson (linear) from Spearman (monotonic rank) correlation.

3. PROJECT REQUIREMENTS (Group Project 4):
   - The task is One-vs-Rest classification for 10 network apps from the Kaggle IP Flows dataset.
   - Classifiers: evaluate Decision Tree, Random Forest/SVM, and MANDATORILY a PyTorch MLP.
   - Differentiable PyTorch model is required for Adversarial Robustness Toolbox (ART) gradient attacks (FGSM, PGD). Output raw logits (no softmax in forward).
   - Normalize features in [0, 1] with MinMaxScaler (fit strictly on train set) for coherence with L2 perturbation norm (||x - x_adv||_2 <= epsilon).
   - Evaluate using Precision, Recall (Class 1), F1-Score (macro/weighted), and ROC-AUC due to class imbalance.
   - In adversarial attacks, compare unconstrained vs domain-constrained attacks (padding >= 0, delay >= 0, mask parameter in ART).
   - Implement Backdoor poisoning and evaluate Defender Clustering (K-Means, DBSCAN/GMM) to isolate malicious flows.
```

---

## 📊 Summary of What the Professor Taught & Key Concepts

### 1. Theory Fundamentals (Lectures 2, 3, 4)
- **Data Exploration (L2):** Statistics on populations ($N$) vs. samples ($N-1$); Kernel Density Estimation (KDE) and bandwidth selection; Empirical Cumulative Distribution Functions (ECDF); Boxplots with Tukey's $1.5 \times \text{IQR}$ rule (accounting for $99.3\%$ of normal distributions, $0.7\%$ outliers); GESD and DBSCAN for outlier identification; Linear Pearson vs. Monotonic Rank Spearman correlation.
- **Data Preprocessing (L3):** Four attribute scales (Nominal, Ordinal, Interval, Ratio); Metric space properties; Minkowski ($L_p$), Mahalanobis (covariance-aware), Cosine (directional), and Jaccard (binary set) distances; Min-Max scaling vs. $Z$-Score standardization; Categorical encoding (One-Hot, Dummy with $k-1$ features to avoid dummy variable trap, Effect coding); Bag-of-Words and TF-IDF text representations.
- **Dimensionality Reduction (L4):** Curse of dimensionality (empty space phenomenon and distance concentration); Spectral theorem and eigendecomposition of sample covariance matrix $\mathbf{S}$; PCA mathematical derivation proving maximization of projected sample variance; Scree plots, Kaiser rule ($\lambda_i > 1$), and cumulative variance thresholds ($80\%-90\%$); Loading scores for feature attribution; Comparison between PCA (linear, invertible, global) and t-SNE (non-linear, stochastic, strictly for 2D/3D visualization).

### 2. Practical Laboratories (Labs 2, 3, 4)
- **Lab 2:** Advanced Pandas slicing and grouping; Matplotlib multi-panel subplots; Pairwise feature scatter plots via `itertools.combinations`; Vectorized KNN classifier from scratch with Scikit-learn API; Subclassing for Minkowski and Cosine distance metrics; Geo-spatial bounding box filtering on New York OpenStreetMap POIs.
- **Lab 3:** Seaborn distribution plotting; Network traffic flow parsing on Darknet packet traces; Unix timestamp resampling into fixed-size temporal bins (`resample('60s').size()`); Network flow definition via 4-tuple `(src_ip, src_port, dst_ip, dst_port)`; Heavy-tailed packet inter-arrival time calculation and log-scale boxplot analysis.
- **Lab 4:** Network feature aggregation per Source IP into a 16-feature host profile matrix; Feature standardization with `StandardScaler`; Pearson and Spearman correlation heatmaps; PCA cumulative explained variance scree plot; Exact Loading Scores calculation; Demonstration of orthogonality in PCA projection space.

### 3. The Course Project (Group Project 4)
- **Topic:** Analyzing Adversarial Attacks on Network Data Classifiers.
- **Goal:** Defend and attack machine learning models classifying network traffic flows into 10 popular applications using tabular flow statistics.
- **Key Techniques:** Stratified sampling, Min-Max normalization, PyTorch Multi-Layer Perceptrons, Adversarial Robustness Toolbox (ART), $L_2$-norm constrained perturbations ($\varepsilon$), domain-specific network traffic constraints (non-negative packet padding and latency delays), Adversarial Training defense, Backdoor poisoning injection, and Defender Unsupervised Clustering (anomaly isolation).

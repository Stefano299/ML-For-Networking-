# ML4N & Data Science: Professor Coding Style & Implementation Patterns

> **Course:** Machine Learning for Networking (ML4N) / Data Science Lab  
> **Institution:** Politecnico di Torino (DBDMG / Telecommunication Networks Group)  
> **Instructor:** Prof. Luca Vassio  
> **Purpose:** Authoritative reference guide documenting the coding style, architectural patterns, vectorized idioms, and evaluation standards enforced by the professor in laboratories and projects.

---

## 1. Technological Stack

| Category | Library / Module | Primary Purpose & Usage |
| :--- | :--- | :--- |
| **Numeric & Tensor Computing** | `numpy` | Vectorization, broadcasting, distance calculation, `np.random.default_rng` |
| **Tabular Data Analysis** | `pandas` | Groupby aggregations, time-series `resample`, `.diff()`, `.copy()` |
| **Machine Learning & Preprocessing**| `scikit-learn` | `StandardScaler`, `MinMaxScaler`, `PCA`, datasets, metric evaluation |
| **Deep Learning & Gradient Attacks**| `torch`, `torch.nn` | Differentiable Multi-Layer Perceptrons (MLP) for ART compatibility |
| **Adversarial Robustness** | `adversarial-robustness-toolbox` (`art`) | `PyTorchClassifier`, FGSM, PGD, Adversarial Training, feature masking |
| **Visualization (OOP Style)** | `matplotlib.pyplot` | Figure/Axes subplots (`fig, ax`), log scales, clean axis labels |
| **Statistical Plotting** | `seaborn` | `sns.histplot(kde=True)`, `sns.ecdfplot`, `sns.heatmap(annot=True, cmap='Blues')` |
| **Standard Library Utilities** | `pathlib.Path`, `itertools` | Clean cross-platform filesystem paths, pairwise combinations (`combinations`) |

---

## 2. Core Programming Principles

### 2.1 NumPy: Strict Vectorization & Broadcasting
The professor strictly prohibits native Python `for` loops across rows or data samples for numerical operations. Operations must leverage array vectorization or multidimensional broadcasting.

1. **Modern Random Number Generator**:
   Avoid legacy `np.random.seed()` or `np.random.permutation()` calls. Use `np.random.default_rng(seed)`:
   ```python
   rng = np.random.default_rng(seed=0)
   indices = rng.permutation(len(X))
   ```

2. **Distance Calculation (Vectorized 1 vs. All)**:
   ```python
   # x: vector (C,), X: matrix (R, C)
   # Broadcasting computes difference across all rows simultaneously
   distances = np.linalg.norm(X - x, axis=1)
   ```

3. **Pairwise Distance Broadcasting (All vs. All)**:
   ```python
   # X: (N, C), X_train: (R, C) -> (N, 1, C) - (R, C) expands to (N, R, C)
   distances = np.linalg.norm(X[:, np.newaxis, :] - X_train, axis=2)  # Output shape: (N, R)
   ```

4. **NumPy Array Conversion**:
   Always use `.to_numpy()` to extract underlying NumPy arrays from Pandas structures. Do not use the deprecated `.values`.
   ```python
   X = df[FEATURES].to_numpy()
   ```

---

### 2.2 Scikit-Learn-Compatible Object-Oriented Design (OOP)
When implementing algorithms from scratch (e.g., $K$-Nearest Neighbors in Lab 2), models must be encapsulated in classes following the Scikit-learn API:

```python
import numpy as np

class KNearestNeighbors:
    def __init__(self, k: int = 3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Stores the training data without explicit computations."""
        self.X_train = X
        self.y_train = y

    def distances(self, x: np.ndarray) -> np.ndarray:
        """Computes Euclidean distance between query sample x and all training points."""
        return np.linalg.norm(self.X_train - x, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vectorized inference with majority voting."""
        if self.X_train is None:
            raise RuntimeError("Must call fit() before predict()")
        
        predictions = []
        for x in X:
            neighbors = np.argsort(self.distances(x))[:self.k]
            labels, counts = np.unique(self.y_train[neighbors], return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.array(predictions)
```

**OOP Inheritance for Custom Distance Metrics**:
Use `super().__init__(k)` and override the `distances()` method to support different metrics:

```python
class MinkowskiKNN(KNearestNeighbors):
    def __init__(self, k: int, p: float):
        super().__init__(k)
        self.p = p

    def distances(self, x: np.ndarray) -> np.ndarray:
        return (np.abs(self.X_train - x) ** self.p).sum(axis=1) ** (1 / self.p)

class CosineKNN(KNearestNeighbors):
    def fit(self, X: np.ndarray, y: np.ndarray):
        super().fit(X, y)
        self.train_norms = np.linalg.norm(X, axis=1)

    def distances(self, x: np.ndarray) -> np.ndarray:
        dot = (self.X_train * x).sum(axis=1)
        return 1.0 - dot / (self.train_norms * np.linalg.norm(x))
```

---

### 2.3 Pandas: Best Practices & Mathematical Nuances

1. **Standard Deviation Degrees of Freedom (`ddof`)**:
   - `pandas.Series.std()` defaults to `ddof=1` (sample standard deviation, divided by $N-1$).
   - `numpy.std()` defaults to `ddof=0` (population standard deviation, divided by $N$).
   - *Requirement:* State `ddof` explicitly when computing descriptive statistics:
     ```python
     sample_std = df.groupby("class").std()         # ddof=1 (sample)
     pop_std = df.groupby("class").std(ddof=0)      # ddof=0 (population)
     ```

2. **Preventing `SettingWithCopyWarning`**:
   Always explicitly `.copy()` subsets of DataFrames before mutating or adding columns:
   ```python
   pois_df = all_pois_df[all_pois_df.index.isin(target_ids)].copy()
   pois_df["category"] = "amenity"
   ```

3. **Index-Based Filtering**:
   Use `df[df.index.isin(id_list)]` instead of chained slicing to avoid unintentional cartesian duplication.

4. **Multi-level Aggregations**:
   ```python
   features_df = df.groupby("src_ip").agg({
       "packets": ["sum"],
       "dst_port": ["nunique", "min", "max", "mean", "std"],
       "pck_len": ["min", "max", "mean", "std"],
       "ttl": ["min", "max", "mean", "std"]
   }).fillna(0)
   # Flatten hierarchical column index:
   features_df.columns = [f"{col}_{stat}" for col, stat in features_df.columns]
   ```

---

### 2.4 Network Traffic Analysis Patterns (Darknet / NetFlows)

1. **Unix Epoch Timestamp Conversion**:
   Convert timestamps in seconds to datetime objects and set as index for time-series operations:
   ```python
   df["ts"] = pd.to_datetime(df["ts"], unit="s")
   df.set_index("ts", inplace=True)
   ```

2. **Temporal Window Resampling**:
   ```python
   num_sec = 60
   packets_per_interval = df.resample(f"{num_sec}s").size()
   ```

3. **Definition of a Network Flow / Link**:
   A bidirectional flow or link is defined by the 4-tuple `(src_ip, src_port, dst_ip, dst_port)`:
   ```python
   links = df.groupby(["src_ip", "src_port", "dst_ip", "dst_port"])
   top_links = links.size().nlargest(10).index
   ```

4. **Inter-Arrival Times (Packet Timing Analysis)**:
   Network packets for a specific link or host must be chronologically sorted before computing differences:
   ```python
   intertimes = df_link["ts"].sort_values().diff()[1:].values
   # Filter out simultaneous packets (intertime == 0)
   intertimes = intertimes[intertimes != 0]
   ```
   *Plotting rule:* Network inter-arrival times exhibit heavy-tailed distributions; always plot them using a **logarithmic scale**:
   ```python
   plt.yscale("log")
   ```

---

### 2.5 Preprocessing, Correlation & PCA Standards

1. **Mandatory Scaling Prior to PCA**:
   Features must be standardized (`StandardScaler`) before Principal Component Analysis:
   ```python
   from sklearn.preprocessing import StandardScaler
   from sklearn.decomposition import PCA

   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```
   *Reasoning:* Features with vast numerical ranges (e.g., port numbers up to 65535 or byte counters) dominate the variance in unscaled data, misleading PCA into ignoring low-magnitude but highly discriminative features (e.g., protocol numbers or flag ratios).

2. **Correlation Heatmaps (Pearson vs. Spearman)**:
   ```python
   corr_p = df_scaled.corr().abs()                      # Linear correlation
   corr_s = df_scaled.corr(method="spearman").abs()      # Monotonic rank correlation

   plt.figure(figsize=(10, 8))
   sns.heatmap(corr_p, cmap="Blues", annot=True, fmt=".2f", vmin=0, vmax=1)
   plt.title("Pearson Correlation Heatmap")
   ```

3. **Scree Plot & Cumulative Explained Variance**:
   ```python
   pca = PCA(random_state=15)
   pca.fit(X_scaled)
   cumul_var = np.cumsum(pca.explained_variance_ratio_) * 100

   plt.plot(range(1, len(cumul_var) + 1), cumul_var, marker="o", color="navy")
   plt.axhline(y=90, color="r", linestyle="--", label="90% Threshold")
   plt.xlabel("Number of Principal Components")
   plt.ylabel("Cumulative Explained Variance (%)")
   ```

4. **Loading Scores Calculation**:
   Calculate feature influence on principal components using the professor's exact formula:
   $$\text{Loading Scores} = \text{pca.components\_}^T \cdot \sqrt{\text{pca.explained\_variance\_}}$$
   ```python
   n_comp = 10
   pca = PCA(n_components=n_comp, random_state=15)
   pca.fit(X_scaled)

   loadings = pd.DataFrame(
       data=pca.components_.T * np.sqrt(pca.explained_variance_),
       columns=[f"PC{i}" for i in range(1, n_comp + 1)],
       index=feature_names
   )
   ```

5. **Orthogonality of PCA Space**:
   Transforming data into PCA space decorrelates features; the covariance matrix of `pca.transform(X_scaled)` is a diagonal matrix (correlation matrix is the identity).

---

### 2.6 Visualization Standards (Matplotlib & Seaborn)

Every plot produced must adhere to publication-grade standards:
- **Object-Oriented Matplotlib:** Prefer `fig, ax = plt.subplots(...)` over procedural `plt.` calls when managing multiple subplots.
- **Explicit Figure Dimensions:** Always provide `figsize=(width, height)`.
- **Descriptive Labels & Units:** Every axis must have a title and explicit units (e.g., `"Time [s]"`, `"Packet Size [bytes]"`, `"Intertime [s] (log scale)"`).
- **Rotated Tick Labels:** When displaying categorical features or link tuples, rotate labels for readability:
  ```python
  plt.xticks(rotation=45, ha="right")
  ```
- **Grid Transparency:** Enable subtle background grids for readability:
  ```python
  plt.grid(True, alpha=0.3)
  ```
- **Scatter Alpha & Size:** For dense scatter plots (e.g., thousands of network flows or POI map coordinates), use small markers with transparency:
  ```python
  ax.scatter(x, y, s=2, alpha=0.5)
  ```

---

## 3. Analytical Interpretation & Typical Exam Questions

The professor consistently tests deep conceptual understanding of why algorithms and statistical choices are made:

1. **Why must data be standardized before applying PCA?**
   PCA seeks orthogonal axes that maximize sample variance. If features are not on a standardized scale ($\mu=0, \sigma=1$), features with naturally high numerical magnitudes (e.g., port numbers up to 65535 or byte counters in millions) will artificially display variance orders of magnitude higher than bounded features (e.g., protocol numbers or packet loss rates). Consequently, the first principal components would simply replicate the high-magnitude features, failing to discover multivariate variance.

2. **What is the difference between Pearson and Spearman correlation?**
   - **Pearson ($r$):** Measures the strength and direction of a strictly *linear* relationship between continuous variables. It is sensitive to outliers and assumes approximate normality.
   - **Spearman ($r_s$):** A non-parametric measure calculated on the *ranks* of the data. It measures monotonic relationships (whether variables increase/decrease together), regardless of whether the relationship is linear. It is robust to outliers and heavy-tailed distributions.

3. **Why must network inter-arrival times be plotted on a logarithmic scale?**
   Network traffic exhibits heavy-tailed, bursty behavior: the vast majority of packets in an active flow arrive within milliseconds (bursts), while idle periods between bursts can last seconds or minutes. On a linear scale, $99\%$ of the distribution would be compressed into a single unreadable bar at the origin. A logarithmic scale stretches the small values and allows simultaneous visualization of millisecond bursts and second-level idle times.

4. **What do Loading Scores represent?**
   Loading scores represent the covariance (and, when standardized, correlation) between original features and the principal components. A high positive or negative loading score on PC1 indicates that the corresponding feature contributes heavily to defining that direction of variance. Features with loading scores close to zero are orthogonal/unrelated to that component.

5. **Why can't t-SNE be used for feature engineering before classification?**
   t-SNE is a non-linear, stochastic, and non-parametric dimensionality reduction technique designed purely for low-dimensional (2D/3D) visualization. It has no explicit projection matrix or mapping function (one cannot project unseen test points into an existing t-SNE map without rerunning the entire optimization), it does not preserve global distances or densities, and its output varies between runs.

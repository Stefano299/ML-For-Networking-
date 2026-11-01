# ML4N Laboratory Reference Implementations

This directory contains clean, verified reference implementations for the practical laboratories of the **Machine Learning for Networking (ML4N)** course at Politecnico di Torino.

All Python scripts are formatted with interactive `# %%` cell blocks (compatible with VS Code, Cursor, and Jupyter).

---

## Laboratory Index

### 1. [Lab #2: Iris Exploration, KNN from Scratch & NYC POIs](lab2_numpy_pandas_knn.py)
- **Topics & Techniques:**
  - Tabular data exploration with Pandas (`groupby`, multi-level aggregations).
  - Difference between sample standard deviation (`ddof=1`) and population standard deviation (`ddof=0`).
  - Matplotlib subplots, boxplots, and pairwise feature combinations via `itertools.combinations`.
  - Pure NumPy vectorized $K$-Nearest Neighbors implementation from scratch (avoiding Python `for` loops with broadcasting).
  - Object-Oriented Machine Learning design (Scikit-Learn style: `__init__`, `fit`, `predict`).
  - Class inheritance with `super().__init__()` for custom distance metrics (Minkowski distance with parameter $p$, Cosine distance).
  - Geospatial tabular filtering on New York City OpenStreetMap Points of Interest (POIs).

### 2. [Lab #3: Data Visualization & Network Traffic Analysis](lab3_visualization_networking.py)
- **Topics & Techniques:**
  - Advanced visualization with Seaborn (`histplot`, `kdeplot`, `ecdfplot`, `boxplot`, `jointplot`).
  - Practical network traffic analysis on the `darknet_traces.csv` dataset.
  - Unix epoch timestamp parsing with `pd.to_datetime(unit='s')` and index manipulation.
  - Regular temporal window resampling (`.resample('60s').size()`) for packet rates.
  - Identification of network links / flows defined as 4-tuples: `(src_ip, src_port, dst_ip, dst_port)`.
  - Chronological sorting and calculation of packet inter-arrival times using `.sort_values().diff()[1:].values`.
  - Heavy-tailed distribution visualization with log-scale boxplots (`plt.yscale('log')`).

### 3. [Lab #4: Preprocessing, Correlation & PCA on Host Profiles](lab4_pca_dimensionality_reduction.py)
- **Topics & Techniques:**
  - Feature engineering on raw packet traces: aggregating network traffic by Source IP (`src_ip`) into a 16-feature host profile matrix.
  - Mandatory feature scaling using `StandardScaler` (demonstrating why unscaled data leads to distorted PCA variance).
  - Correlation analysis: Pearson linear correlation vs. Spearman monotonic rank correlation via `sns.heatmap`.
  - Principal Component Analysis (PCA) using `sklearn.decomposition.PCA`.
  - Scree plot generation and evaluation of cumulative explained variance (`np.cumsum(pca.explained_variance_ratio_)`).
  - Calculation and bar plot visualization of **Loading Scores** using the course formula:
    $$\text{Loading Scores} = \text{pca.components\_}^T \cdot \sqrt{\text{pca.explained\_variance\_}}$$
  - Verification of orthogonality in PCA projection space (diagonal covariance / identity correlation).

---

## Coding Patterns Quick Reference

When developing any module or notebook, ensure you align with the patterns in these scripts. Refer to [`../CODING_STYLE_GUIDE.md`](../CODING_STYLE_GUIDE.md) for full architectural guidelines.

"""
Lab #4 - Preprocessing, Correlation Analysis and Dimensionality Reduction (PCA)
Politecnico di Torino - Machine Learning for Networking / Data Science Lab

This file is formatted with '# %%' interactive cell markers for VS Code / Cursor / Jupyter.
It covers:
1. Standardization with StandardScaler
2. Correlation Analysis: Pearson vs Spearman rank correlation
3. Principal Component Analysis (PCA): fit, transform, cumulative explained variance
4. Loading Scores calculation: pca.components_.T * np.sqrt(pca.explained_variance_)
5. Network Feature Engineering: aggregating packet traces into host-level (Source IP) features
"""

# %% [markdown]
# # Lab #4: Preprocessing and Dimensionality Reduction
#
# Objectives:
# 1. Feature scaling with `StandardScaler`.
# 2. Linear (Pearson) and non-linear monotonic (Spearman) correlation analysis with `sns.heatmap`.
# 3. Principal Component Analysis (PCA) with Scikit-learn.
# 4. Scree plots / Cumulative Explained Variance evaluation.
# 5. Feature Loading Score computation and interpretation.
# 6. Host-level Network Traffic Characterization from packet traces (`darknet_traces.csv`).

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## 1. Tutorial: Data Scaling, Correlation, and PCA on Iris Dataset

# %% [markdown]
# ### 1.1 Data Scaling with StandardScaler
# `StandardScaler` removes the mean and scales features to unit variance (z-score normalization):
# $$z = \frac{x - \mu}{\sigma}$$

# %%
iris_data = datasets.load_iris()
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
df_iris = pd.DataFrame(iris_data.data, columns=columns)
df_iris["type"] = "setosa"
df_iris.loc[50:99, "type"] = "versicolor"
df_iris.loc[100:149, "type"] = "virginica"

# Scaling features
scaler = StandardScaler()
df_iris_scaled = df_iris.copy()
df_iris_scaled[columns] = scaler.fit_transform(df_iris[columns])

print("Original means:", df_iris[columns].mean().values.round(2))
print("Scaled means (approx 0):", df_iris_scaled[columns].mean().values.round(2))
print("Scaled stds  (approx 1):", df_iris_scaled[columns].std(ddof=0).values.round(2))

# %% [markdown]
# ### 1.2 Correlation Analysis (Pearson Heatmap)
# Pearson correlation coefficient evaluates linear relationships:
# $$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$

# %%
correlation_matrix = df_iris_scaled[columns].corr().abs()

plt.figure(figsize=(6, 5))
sns.heatmap(
    correlation_matrix,
    cmap="Blues",
    annot=True,
    fmt=".2f",
    vmin=0,
    vmax=1,
    cbar_kws={"label": "Absolute Correlation"}
)
plt.title("Iris Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.3 Principal Component Analysis (PCA) & Cumulative Explained Variance
# Scree plot determines how many components are required to retain sufficient variance (e.g. 90-95%).

# %%
pca = PCA(random_state=15)
pca.fit(df_iris_scaled[columns])

explained_variance = pca.explained_variance_ratio_
cumul_exp_var = np.cumsum(explained_variance) * 100

print("Explained variance ratio per PC:", np.round(explained_variance, 4))
print("Cumulative explained variance (%):", np.round(cumul_exp_var, 2))

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(cumul_exp_var) + 1), cumul_exp_var, marker="o", linestyle="-", color="b")
plt.axhline(y=95, color="r", linestyle="--", label="95% Threshold")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("Iris PCA Explained Variance")
plt.xticks(range(1, len(cumul_exp_var) + 1))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.4 PCA Transformation (Projection)

# %%
pca_3 = PCA(n_components=3, random_state=15)
pca_result = pca_3.fit_transform(df_iris_scaled[columns])

df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2", "PC3"])
df_pca["type"] = df_iris["type"]

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="type", s=50)
plt.title("Iris Projected onto First 2 Principal Components")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.5 Loading Scores Calculation
# Formula:
# $$\text{Loading Scores} = \text{pca.components\_}^T \cdot \sqrt{\text{pca.explained\_variance\_}}$$
# Loading scores measure the weight and contribution of each original feature to each principal component.

# %%
loadings = pd.DataFrame(
    data=pca_3.components_.T * np.sqrt(pca_3.explained_variance_),
    columns=["PC1", "PC2", "PC3"],
    index=columns
)
print("Loading Scores:")
print(loadings.round(4))

# Visualize loading scores on PC1 and PC2
plt.figure(figsize=(8, 4))
loadings[["PC1", "PC2"]].sort_values("PC1").plot(kind="bar", figsize=(8, 4))
plt.title("Feature Loading Scores for PC1 and PC2")
plt.ylabel("Loading Score")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 2. Exercise: Darknet Traffic Feature Engineering & Dimensionality Reduction

# %% [markdown]
# ### 2.1 Dataset Management & Feature Extraction per Source IP
# We convert packet-level traces into host-level profiles (1 row per distinct `src_ip`)
# with 16 aggregated features:
# - `packets_sum`: Total packets sent
# - `dst_port_nunique`: Number of distinct ports contacted
# - `proto_nunique`: Number of distinct protocols used
# - `dst_ip_nunique`: Number of distinct destination IPs contacted
# - `dst_port_[min, max, mean, std]`: Statistics of contacted destination ports
# - `pck_len_[min, max, mean, std]`: Statistics of packet sizes
# - `ttl_[min, max, mean, std]`: Statistics of Time-to-Live

# %%
DATASET_PATH = Path("darknet_traces.csv")
if not DATASET_PATH.exists():
    DATASET_PATH = Path("../dataset.csv")

if DATASET_PATH.exists():
    df = pd.read_csv(DATASET_PATH, nrows=300000)
else:
    # Synthetic generator if CSV not found locally
    rng = np.random.default_rng(42)
    n = 20000
    df = pd.DataFrame({
        "ts": 1570000000 + np.sort(rng.integers(0, 7200, n)),
        "src_ip": rng.choice([f"192.168.1.{i}" for i in range(1, 100)], n),
        "src_port": rng.integers(1024, 65535, n),
        "dst_ip": rng.choice(["10.0.0.1", "10.0.0.2", "10.0.0.3"], n),
        "dst_port": rng.integers(1, 65535, n),
        "proto": rng.choice([6, 17, 1], n),
        "pck_len": rng.integers(40, 1500, n),
        "ttl": rng.integers(30, 128, n)
    })

df["packets"] = 1

# Aggregate 16 features per source IP
agg_dict = {
    "packets": ["sum"],
    "dst_port": ["nunique", "min", "max", "mean", "std"],
    "proto": ["nunique"],
    "dst_ip": ["nunique"],
    "pck_len": ["min", "max", "mean", "std"],
    "ttl": ["min", "max", "mean", "std"]
}

features_df = df.groupby("src_ip").agg(agg_dict)
# Flatten hierarchical multi-index column names
features_df.columns = [f"{col}_{stat}" for col, stat in features_df.columns]
# Standard deviation of a single packet is NaN; fill with 0
features_df = features_df.fillna(0)

print("Aggregated Features DataFrame shape:", features_df.shape)
print("Extracted 16 features:", list(features_df.columns))

# %% [markdown]
# ### Standardization & Visual Impact Comparison
# Standardization centers data around 0 with unit variance without altering relative geometric relationships.

# %%
scaler_dark = StandardScaler()
features_scaled = pd.DataFrame(
    scaler_dark.fit_transform(features_df),
    columns=features_df.columns,
    index=features_df.index
)

# Random subset of 1000 hosts (or all if < 1000)
sample_size = min(1000, len(features_df))
sample_indices = np.random.default_rng(42).choice(features_df.index, size=sample_size, replace=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Raw features
axes[0].scatter(
    features_df.loc[sample_indices, "packets_sum"],
    features_df.loc[sample_indices, "dst_port_nunique"],
    alpha=0.6,
    color="steelblue"
)
axes[0].set_title("Raw Features: Packets vs Contacted Ports")
axes[0].set_xlabel("Packets Sent")
axes[0].set_ylabel("Distinct Destination Ports")
axes[0].grid(True, alpha=0.3)

# Plot 2: Standardized features
axes[1].scatter(
    features_scaled.loc[sample_indices, "packets_sum"],
    features_scaled.loc[sample_indices, "dst_port_nunique"],
    alpha=0.6,
    color="coral"
)
axes[1].set_title("Standardized Features (Zero-mean, Unit-variance)")
axes[1].set_xlabel("Standardized Packets Sent")
axes[1].set_ylabel("Standardized Distinct Ports")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.2 Correlation Analysis (Pearson Heatmap)
# Identifies collinear features (e.g., min/max/mean of TTL or port numbers).

# %%
corr_matrix = features_scaled.corr().abs()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="Blues", annot=True, fmt=".2f", vmin=0, vmax=1)
plt.title("Correlation Matrix of Darknet Host Features (Pearson)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.3 PCA on Standardized Features & Cumulative Variance

# %%
pca_dark = PCA(random_state=15)
pca_dark.fit(features_scaled)

cumul_var = np.cumsum(pca_dark.explained_variance_ratio_) * 100

plt.figure(figsize=(9, 4))
plt.plot(range(1, len(cumul_var) + 1), cumul_var, marker="o", color="darkblue")
plt.axhline(y=90, color="r", linestyle="--", label="90% Variance Threshold")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("Darknet Host Profile - PCA Explained Variance")
plt.xticks(range(1, len(cumul_var) + 1))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Loading Scores on PC1 and PC2 (Top 10 Components Retained)

# %%
n_components_chosen = min(10, features_scaled.shape[1])
pca_10 = PCA(n_components=n_components_chosen, random_state=15)
pca_10.fit(features_scaled)

dark_loadings = pd.DataFrame(
    data=pca_10.components_.T * np.sqrt(pca_10.explained_variance_),
    columns=[f"PC{i}" for i in range(1, n_components_chosen + 1)],
    index=features_scaled.columns
)

plt.figure(figsize=(14, 5))
dark_loadings[["PC1", "PC2"]].plot(kind="bar", figsize=(14, 5))
plt.title("Feature Loading Scores for PC1 and PC2 (Darknet Dataset)")
plt.ylabel("Loading Score")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.4 Correlation Heatmap After PCA Transformation
# Demonstrates orthogonality: principal components are completely uncorrelated (off-diagonal correlations = 0).

# %%
pca_transformed = pca_10.transform(features_scaled)
df_pca_components = pd.DataFrame(
    pca_transformed,
    columns=[f"PC{i}" for i in range(1, n_components_chosen + 1)]
)

plt.figure(figsize=(8, 6))
sns.heatmap(df_pca_components.corr().abs(), cmap="Blues", annot=True, fmt=".2f", vmin=0, vmax=1)
plt.title("Correlation Matrix of Principal Components (Orthogonal Space)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.5 PCA Without Standardization (The Pitfall)
# Demonstrates why standardization is critical: unscaled features with large raw numbers (e.g. port numbers up to 65535)
# dominate 99% of variance, hiding subtle structural variations in smaller magnitude features.

# %%
pca_unscaled = PCA(random_state=15)
pca_unscaled.fit(features_df)
cumul_var_unscaled = np.cumsum(pca_unscaled.explained_variance_ratio_) * 100

print("Unscaled Explained Variance of first component alone: {:.2f}%".format(cumul_var_unscaled[0]))

# %% [markdown]
# ### 2.6 Spearman Rank Correlation Coefficient
# Uses `method='spearman'` to detect monotonic non-linear dependencies.

# %%
corr_spearman = features_scaled.corr(method="spearman").abs()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_spearman, cmap="Blues", annot=True, fmt=".2f", vmin=0, vmax=1)
plt.title("Spearman Rank Correlation Matrix of Darknet Host Features")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

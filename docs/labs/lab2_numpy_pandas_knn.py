"""
Lab #2 - NumPy, pandas and Matplotlib: Iris Exploration, KNN from Scratch, NYC POIs
Politecnico di Torino - Machine Learning for Networking / Data Science Lab

This file is formatted with '# %%' interactive cell markers for VS Code / Cursor / Jupyter.
It contains the clean solutions, vectorized patterns, and OOP design taught by the professor.
"""

# %% [markdown]
# # Lab #2: NumPy, pandas and Matplotlib
# Topics covered:
# - Exploring tabular data with pandas (grouping, aggregation, summary statistics)
# - Matplotlib subplots, boxplots, and pairwise scatter plots
# - Vectorization and broadcasting in NumPy (implementing KNN from scratch)
# - Object-Oriented Machine Learning APIs (fit / predict, OOP inheritance for distance metrics)
# - Geospatial / Tabular data processing on New York City POIs dataset

# %%
import math
import urllib.request
import zipfile
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("NumPy", np.__version__)
print("pandas", pd.__version__)
print("Matplotlib", matplotlib.__version__)

# %% [markdown]
# ### 0.2 Datasets Setup
# In this lab two datasets are used:
# 1. Iris dataset (UCI ML Repository)
# 2. NYC Points of Interest (OpenStreetMap extract)

# %%
IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
POIS_URL = "https://github.com/dbdmg/data-science-lab/raw/master/datasets/NYC_POIs.zip"

if not Path("iris.csv").exists():
    urllib.request.urlretrieve(IRIS_URL, "iris.csv")

if not Path("pois_all_info").exists():
    urllib.request.urlretrieve(POIS_URL, "NYC_POIs.zip")
    with zipfile.ZipFile("NYC_POIs.zip") as archive:
        for name in archive.namelist():
            if not name.startswith("__MACOSX"):  # skip macOS metadata
                archive.extract(name)

# %% [markdown]
# ## 1. Iris: From Exploration to a Classifier

# %% [markdown]
# ### 1.1 Load the dataset
# Features: sepal_length, sepal_width, petal_length, petal_width, class

# %%
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

iris_df = pd.read_csv("iris.csv", header=None, names=FEATURES + ["class"])
print("Shape:", iris_df.shape)
iris_df.head()

# %% [markdown]
# ### 1.2 Statistics per species
# Note on degrees of freedom:
# - pandas computes sample standard deviation (ddof=1, dividing by n-1).
# - numpy computes population standard deviation by default (ddof=0, dividing by n).
# - std(ddof=0) reproduces population standard deviation.

# %%
print("Sample std (default ddof=1):")
print(iris_df.groupby("class").agg(["mean", "std"]))

print("\nPopulation std (ddof=0):")
print(iris_df.groupby("class").std(ddof=0))

# %% [markdown]
# ### 1.3 Distribution of the features (Boxplot)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot([iris_df[feature] for feature in FEATURES], tick_labels=FEATURES)
ax.set_ylabel("cm")
ax.set_title("Iris Feature Distributions")
plt.show()

# %% [markdown]
# ### 1.4 Distribution of features per species

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (species, species_df) in zip(axes, iris_df.groupby("class")):
    ax.boxplot([species_df[feature] for feature in FEATURES], tick_labels=FEATURES)
    ax.set_title(species)
axes[0].set_ylabel("cm")
plt.show()

# %% [markdown]
# ### 1.5 Pairs of features (Pairwise Scatter Plot)
# C(4, 2) = 6 feature pairs

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (feature_x, feature_y) in zip(axes.flat, combinations(FEATURES, 2)):
    for species, species_df in iris_df.groupby("class"):
        ax.scatter(species_df[feature_x], species_df[feature_y], s=10, label=species)
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.legend(fontsize="small")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 1.6 From pandas to NumPy
# Converting features and target labels into clean NumPy ndarrays.

# %%
SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

X = iris_df[FEATURES].to_numpy()
y = iris_df["class"].map({species: label for label, species in enumerate(SPECIES)}).to_numpy()
print("X shape, dtype:", X.shape, X.dtype)
print("y shape, dtype, class counts:", y.shape, y.dtype, np.bincount(y))

# %% [markdown]
# ### 1.7 Nearest neighbors of a sample (Vectorized function)
# Subtracting x from every row of X by broadcasting, then np.argsort to get indices.

# %%
def k_nearest_neighbors(x, X, k):
    distances = np.linalg.norm(X - x, axis=1)  # x is subtracted from every row of X by broadcasting
    return np.argsort(distances)[:k]

sample_idx = 35
neighbors = k_nearest_neighbors(X[sample_idx], X, k=5)
print(f"Indices of 5 nearest neighbors for sample {sample_idx}:", neighbors)
print("Neighbor labels:", y[neighbors])
print("True label:", y[sample_idx])

# %% [markdown]
# ### 1.8 Train / Test split (Random Permutation)
# The dataset is sorted by class; random shuffling is required before splitting.

# %%
rng = np.random.default_rng(seed=0)
indices = rng.permutation(len(X))  # random permutation of [0, ..., 149]
n_test = len(X) * 20 // 100         # 20% test set

X_test, y_test = X[indices[:n_test]], y[indices[:n_test]]
X_train, y_train = X[indices[n_test:]], y[indices[n_test:]]

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# %% [markdown]
# ### 1.9 KNN as a Class (Scikit-Learn API style)
# Object-Oriented implementation using fit() and predict().

# %%
class KNearestNeighbors:
    def __init__(self, k):
        """
        Store k and initialize training data attributes.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store prior training data.
        X: shape (R, C), y: shape (R,)
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Classify samples in X.
        X: shape (N, C) -> returns labels of shape (N,)
        """
        if self.X_train is None:
            raise RuntimeError("call fit() before predict()")
        predictions = []
        for x in X:
            neighbors = k_nearest_neighbors(x, self.X_train, self.k)
            labels, counts = np.unique(self.y_train[neighbors], return_counts=True)
            predictions.append(labels[np.argmax(counts)])  # majority voting
        return np.array(predictions)

# Fully broadcasted alternative: all pairwise distances at once
class KNearestNeighborsBroadcast(KNearestNeighbors):
    def predict(self, X):
        if self.X_train is None:
            raise RuntimeError("call fit() before predict()")
        # (N, 1, C) - (R, C) -> (N, R, C): difference between each row of X and each training sample
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.X_train, axis=2)  # shape (N, R)
        neighbors = np.argsort(distances, axis=1)[:, : self.k]                 # shape (N, k)
        neighbor_labels = self.y_train[neighbors]                             # shape (N, k)
        predictions = []
        for labels_of_x in neighbor_labels:
            labels, counts = np.unique(labels_of_x, return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.array(predictions)

# %% [markdown]
# ### 1.10 Model Evaluation & Accuracy

# %%
knn = KNearestNeighbors(k=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = (y_pred == y_test).mean()
print("Predicted labels:", y_pred)
print("True labels:     ", y_test)
print(f"Accuracy: {accuracy:.4f}")

# Verify broadcasted version
knn_broadcast = KNearestNeighborsBroadcast(k=3)
knn_broadcast.fit(X_train, y_train)
print("Same predictions between implementations:", (knn_broadcast.predict(X_test) == y_pred).all())

# %% [markdown]
# ### 1.11 Analyzing Prediction Errors

# %%
mistakes = y_pred != y_test
print(f"Number of misclassifications: {mistakes.sum()}")

fig, ax = plt.subplots(figsize=(7, 5))
for species_idx, species in enumerate(SPECIES):
    ax.scatter(
        X_test[y_test == species_idx, 2],
        X_test[y_test == species_idx, 3],
        s=15,
        label=species,
    )
ax.scatter(
    X_test[mistakes, 2],
    X_test[mistakes, 3],
    facecolors="none",
    edgecolors="red",
    s=80,
    linewidths=2,
    label="Mistakes",
)
ax.set_xlabel("petal length (cm)")
ax.set_ylabel("petal width (cm)")
ax.legend()
plt.title("KNN Classification Errors")
plt.show()

# %% [markdown]
# ### 1.12 OOP Inheritance: Alternative Distance Metrics (Minkowski & Cosine)

# %%
class KNearestNeighbors:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def distances(self, x):
        """Euclidean distance between x and each training sample, shape (R,)."""
        return np.linalg.norm(self.X_train - x, axis=1)

    def predict(self, X):
        if self.X_train is None:
            raise RuntimeError("call fit() before predict()")
        predictions = []
        for x in X:
            neighbors = np.argsort(self.distances(x))[: self.k]
            labels, counts = np.unique(self.y_train[neighbors], return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.array(predictions)


class MinkowskiKNN(KNearestNeighbors):
    def __init__(self, k, p):
        super().__init__(k)  # parent initializes k and data storage
        self.p = p

    def distances(self, x):
        return (np.abs(self.X_train - x) ** self.p).sum(axis=1) ** (1 / self.p)


class CosineKNN(KNearestNeighbors):
    def fit(self, X, y):
        super().fit(X, y)
        self.train_norms = np.linalg.norm(X, axis=1)  # precompute norms for efficiency

    def distances(self, x):
        dot = (self.X_train * x).sum(axis=1)
        return 1 - dot / (self.train_norms * np.linalg.norm(x))


models = {
    "euclidean": KNearestNeighbors(k=3),
    "Minkowski, p=1 (Manhattan)": MinkowskiKNN(k=3, p=1),
    "Minkowski, p=3": MinkowskiKNN(k=3, p=3),
    "cosine": CosineKNN(k=3),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = (model.predict(X_test) == y_test).mean()
    print(f"{name:30s}: accuracy {acc:.2f}")

# %% [markdown]
# ---
# ## 2. New York Points of Interest (Tabular & Geospatial Pandas)

# %% [markdown]
# ### 2.1 Load the dataset
# Tab-separated file with mixed types; low_memory=False is used to prevent mixed-type warnings.

# %%
TEXT_COLUMNS = ["amenity", "name", "shop", "public_transport", "highway"]

if Path("pois_all_info").exists():
    all_pois_df = pd.read_csv(
        "pois_all_info",
        sep="\t",
        index_col="@id",
        dtype={column: "string" for column in TEXT_COLUMNS},
        low_memory=False,
    )
    print("All POIs shape:", all_pois_df.shape)

# %% [markdown]
# ### 2.2 Geospatial Bounding Box Filtering

# %%
MAP_EXTENT = [-74.258, -73.7, 40.49, 40.92]  # [lon_min, lon_max, lat_min, lat_max]

if Path("pois_all_info").exists():
    in_box = (
        (all_pois_df["@lon"] >= MAP_EXTENT[0])
        & (all_pois_df["@lon"] <= MAP_EXTENT[1])
        & (all_pois_df["@lat"] >= MAP_EXTENT[2])
        & (all_pois_df["@lat"] <= MAP_EXTENT[3])
    )
    all_pois_df = all_pois_df[in_box]
    print("POIs within bounding box:", len(all_pois_df))

# %% [markdown]
# ### 2.3 Filter NYC Municipality POIs using isin()
# Boolean indexing with .isin() ensures each row appears at most once, avoiding duplication from .loc on duplicated IDs.

# %%
if Path("ny_municipality_pois_id.csv").exists():
    ny_ids = pd.read_csv("ny_municipality_pois_id.csv", header=None, names=["@id"])
    pois_df = all_pois_df[all_pois_df.index.isin(ny_ids["@id"])].copy()
    print(f"{len(pois_df)} of {len(all_pois_df)} POIs are inside NYC municipality")

# %% [markdown]
# ### 2.4 Category Assignment Loop
# Resolving POIs having multiple categories by prioritizing in defined order.

# %%
CATEGORIES = ["amenity", "shop", "public_transport", "highway"]

if "pois_df" in locals():
    pois_df["category"] = None
    for category in CATEGORIES:
        pois_df.loc[pois_df["category"].isna() & pois_df[category].notna(), "category"] = category

    print("POIs with >1 category:", (pois_df[CATEGORIES].notna().sum(axis=1) > 1).sum())
    print("\nCategory distribution:")
    print(pois_df["category"].value_counts(dropna=False))

# %% [markdown]
# ### 2.5 Plotting Category Counts & Top Amenities

# %%
if "pois_df" in locals():
    # Value counts & bar plot
    category_counts = pois_df["category"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(category_counts.index, category_counts.to_numpy())
    ax.set_title("POIs per category")
    ax.set_ylabel("Number of POIs")
    plt.show()

    # Most frequent amenities (value_counts vs groupby.size)
    amenity_counts = pois_df["amenity"].value_counts()
    print("Top 10 amenities:")
    print(amenity_counts.head(10))

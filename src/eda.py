import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split


def feature_label(feature):
    # Axis label with the unit of measurement, deduced from the CICFlowMeter feature name
    # (durations, inter-arrival times and active/idle periods are in microseconds)
    name = feature.lower()
    if name.endswith("bytes.s"):
        unit = "bytes/s"
    elif name.endswith("packets.s"):
        unit = "packets/s"
    elif "variance" in name:
        unit = "bytes²"
    elif any(k in name for k in ("iat", "duration", "active", "idle")):
        unit = "µs"
    elif any(k in name for k in ("length", "size", "bytes")):
        unit = "bytes"
    elif any(k in name for k in ("packets", "pkt")):
        unit = "packets"
    else:
        return feature
    return f"{feature} [{unit}]"


def summary_statistics(X):
    # Per-feature statistics used to describe the shape of the distributions
    return pd.DataFrame({
        "zeros_%": (X == 0).mean() * 100,
        "mean": X.mean(),
        "median": X.median(),
        "p99": X.quantile(0.99),
        "max": X.max(),
        "skewness": X.skew(),
        "tukey_outliers_%": tukey_outlier_share(X) * 100,
        "tukey_outliers_log_%": tukey_outlier_share(np.log1p(X)) * 100,
    })


def tukey_outlier_share(X):
    # Share of values outside the Tukey fences [Q1 - 1.5 IQR, Q3 + 1.5 IQR], for each column
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    outside = (X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)
    return outside.mean()


def univariate_auc(X, y, apps):
    # Simple AUC to see if a feature is a good predictor of the class
    ranks = rankdata(X.to_numpy(), axis=0)  # ties get the average rank
    Y = (y.to_numpy()[:, np.newaxis] == np.array(apps)).astype(float)  # (n_flows, n_apps), one 1 per row
    n1 = Y.sum(axis=0)
    n0 = len(Y) - n1
    auc = (ranks.T @ Y - n1 * (n1 + 1) / 2) / (n1 * n0)
    return pd.DataFrame(auc, index=X.columns, columns=apps)


def mutual_information(X, y, n_samples=100_000, seed=42):
    # Mutual information between each feature and the app label (10 classes)
    # The k-NN estimator of scikit-learn is slow, so I stratify the sample first
    # log1p compresses the heavy tails, so the distances between neighbors are not dominated by a few huge values
    X_s, _, y_s, _ = train_test_split(X, y, train_size=n_samples, stratify=y, random_state=seed)
    is_binary = X_s.isin([0, 1]).all()
    X_s = X_s.astype(float)
    X_s.loc[:, ~is_binary] = np.log1p(X_s.loc[:, ~is_binary])  # 0/1 features stay discrete
    mi = mutual_info_classif(X_s, y_s, discrete_features=is_binary.to_numpy(), random_state=seed)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def exact_vs_range_mi(X, y, n_bins=20):
    # Mutual information with the app label computed in two ways:
    # - exact values: every distinct value is a category, so the specific values can be used
    # - ranges: the feature is cut into n_bins quantile bins, so only the order of the values is used
    #   (equal values always fall in the same bin, so with many ties some bins merge and fewer remain)
    # Both are corrected for the positive bias of the plug-in estimator, (K - 1)(C - 1) / 2N nats (Miller-Madow),
    # which grows with the number K of distinct values
    n_classes = y.nunique()
    rows = {}
    for feature in X.columns:
        values = pd.factorize(X[feature])[0]
        bins = pd.qcut(X[feature], n_bins, labels=False, duplicates="drop")
        n_values = X[feature].nunique()
        n_used_bins = bins.nunique()
        rows[feature] = {
            "distinct_values": n_values,
            "mi_exact_values": mutual_info_score(y, values) - (n_values - 1) * (n_classes - 1) / (2 * len(y)),
            "mi_quantile_bins": mutual_info_score(y, bins) - (n_used_bins - 1) * (n_classes - 1) / (2 * len(y)),
        }
    return pd.DataFrame(rows).T


def correlation_pairs(**corrs):
    # One row for each pair of features, one column for each correlation matrix passed
    first = next(iter(corrs.values()))
    iu = np.triu_indices(len(first), k=1)
    pairs = pd.DataFrame({"feature_1": first.index[iu[0]], "feature_2": first.columns[iu[1]]})
    for name, corr in corrs.items():
        pairs[name] = corr.to_numpy()[iu]
    return pairs


def feature_clusters(corr, threshold=0.9):
    # Hierarchical clustering of the features with distance 1 - |corr|
    # inside each cluster every pair of features has |corr| >= threshold
    # The result follows the order of the dendrogram leaves, so correlated features are next to each other, so they are better displayed inside the heatmap
    dist = 1 - corr.abs().to_numpy()
    np.fill_diagonal(dist, 0)  # 1 - |corr| on the diagonal can be a tiny float instead of exactly 0
    Z = linkage(squareform(dist, checks=False), method="complete")
    clusters = pd.Series(fcluster(Z, t=1 - threshold, criterion="distance"), index=corr.index)
    return clusters.iloc[leaves_list(Z)]


def select_non_redundant(ranking, corr, threshold=0.9):
    # It goes through the features from the most to the least relevant and keeps one
    # only if its absolute correlation with all the features is below the threshold
    selected = []
    for feature in ranking:
        if (corr.loc[feature, selected].abs() < threshold).all():
            selected.append(feature)
    return selected

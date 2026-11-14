import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler

# The 10 target applications requested by the project
APPS = [
    "YOUTUBE", "AMAZON", "GMAIL", "WINDOWS_UPDATE", "MICROSOFT",
    "FACEBOOK", "DROPBOX", "SKYPE", "TWITTER", "CLOUDFLARE"
]

# Service ports for One-Hot Encoding
SERVICE_PORTS = [80, 443, 3128]
# These 3 well-known ports account for ~93.8% of all traffic in the dataset:
# - 3128 (Squid Proxy): 40.08%
# - 443 (HTTPS): 35.07%
# - 80 (HTTP): 18.61%
# All remaining destination ports are grouped into 'Other'.

# Features that the EDA (Section 1.3) found to be copies of other features.
# They are still in model_ready.csv, load_model_ready() removes them
REDUNDANT_FEATURES = [
    "Avg.Fwd.Segment.Size",    # = Fwd.Packet.Length.Mean (up to float rounding)
    "Avg.Bwd.Segment.Size",    # = Bwd.Packet.Length.Mean (up to float rounding)
    "Packet.Length.Variance",  # = Packet.Length.Std^2
]


def load_dataset(path="data/processed/dataset_10apps.csv"):
    return pd.read_csv(path, low_memory=False)


def load_model_ready(path="data/processed/model_ready.csv", groups_path="data/processed/groups.csv"):
    # Loads X, y and groups saved by the preprocessing, without the redundant features.
    # This is used for section 2-4
    data = pd.read_csv(path)
    y = data["label"]
    X = data.drop(columns=["label"] + REDUNDANT_FEATURES)
    groups = pd.read_csv(groups_path)["Flow.ID"]
    return X, y, groups


def filter_10_apps(df):
    mask = df["ProtocolName"].isin(APPS)
    return df[mask].copy()


def get_class_distribution(df):
    counts = df["ProtocolName"].value_counts()
    percentages = (counts / len(df) * 100).round(2)
    return pd.DataFrame({"count": counts, "percentage_%": percentages})


def get_constant_columns(df):
    # Returns column with only one value
    unique_counts = df.nunique()
    return unique_counts[unique_counts <= 1].index.tolist()


def get_duplicate_columns(df):
    # Returns columns that have identical values to earlier columns 
    duplicates = []
    cols = df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            # To compare the colums I don't use df[cols[i]].equals(df[cols[j]]) because it would compare also the types
            if cols[j] not in duplicates and np.array_equal(df[cols[i]].to_numpy(), df[cols[j]].to_numpy()):
                duplicates.append(cols[j])
    return duplicates


def drop_duplicate_rows(df):
    # Removes rows identical in every column (same Flow.ID and Timestamp too)
    return df.drop_duplicates().reset_index(drop=True)


def fix_init_win_sentinel(X):
    # CICFlowMeter writes -1 when the initial TCP window cannot be observed.
    # Forward: -1 only for UDP flows, replaced with 0.
    # Backward: -1 when the server never sent packets (or UDP). 0 is also a real window value,
    # so we add a flag not to lose information
    X = X.copy()
    X["Init_Win_bytes_backward_missing"] = (X["Init_Win_bytes_backward"] == -1).astype(int)
    X["Init_Win_bytes_forward"] = X["Init_Win_bytes_forward"].replace(-1, 0)
    X["Init_Win_bytes_backward"] = X["Init_Win_bytes_backward"].replace(-1, 0)
    return X


def split_by_flow(X, y, groups, test_size=0.2, seed=42):
    # Train/test split stratified on the app label and grouped by Flow.ID:
    # all the rows of the same flow end up on the same side.
    # StratifiedGroupKFold with 1/test_size folds, the first fold is used as test set.
    sgkf = StratifiedGroupKFold(n_splits=round(1 / test_size), shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(X, y, groups))
    return (X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx],
            groups.iloc[train_idx], groups.iloc[test_idx])


def get_continuous_columns(X):
    # Function that returns continuoos (not 0/1) features
    is_binary = X.isin([0, 1]).all()
    return is_binary[~is_binary].index.tolist()


class LogMinMaxScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    # Our normalization as a scikit-learn transformer, so it can go inside a Pipeline and be fitted
    # again in every cross-validation fold (only on the training part of the fold):
    # 1. log1p on the continuous features (the 0/1 columns are left as they are),
    #    because some features have very large values
    # 2. Min-Max to [0, 1], clip=True so values outside the training range stay in [0, 1] (needed by ART)

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)  # for get_feature_names_out
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        # continuous = not only 0/1 values in the data used for the fit
        self.continuous_ = ~np.isin(X, [0, 1]).all(axis=0)
        self.minmax_ = MinMaxScaler(feature_range=(0, 1), clip=True).fit(self._log(X))
        return self

    def transform(self, X):
        return self.minmax_.transform(self._log(np.asarray(X, dtype=float)))

    def _log(self, X):
        X = X.copy()
        X[:, self.continuous_] = np.log1p(X[:, self.continuous_])
        return X


def scale_features(X_train, X_test):
    # LogMinMaxScaler fitted on X_train and applied to both, with the results as DataFrames.
    # Returns the inner MinMaxScaler (data_range_ etc. refer to the log1p values).
    scaler = LogMinMaxScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, scaler.minmax_


def load_train_test(test_size=0.2, seed=42):
    # Same split for everyone in Section 2-4: load model_ready.csv and split by Flow.ID (80/20).
    # X is NOT scaled here: in cross-validation the scaler must be fitted only on the training part of
    # each fold, so it goes inside the model (Pipeline with LogMinMaxScaler, see src/evaluation.py).
    # For the final model (and Section 3-4) fit LogMinMaxScaler on the whole X_train.
    # y_train / y_test are the app names: use make_ovr_target(y_train, app) to get the 0/1 label.
    X, y, groups = load_model_ready()
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_by_flow(X, y, groups, test_size, seed)
    return X_train, X_test, y_train, y_test, groups_train


def make_ovr_target(y, app):
    if app not in APPS:
        raise ValueError(f"App {app} not valid, it must be in APPS")
    return (y == app).astype(int)


def prepare_dataset_for_model(df):
    """
    Prepares the dataset for model training:
    - Removes rows identical in every column
    - Preserves groups = df["Flow.ID"].copy() for StratifiedGroupKFold
    - Removes identifiers: Flow.ID, Source.IP, Destination.IP, Timestamp, Source.Port
    - Separates target label (y = ProtocolName) and removes L7Protocol from features
    - Removes constant columns (only 1 unique value)
    - Removes duplicate columns (identical values)
    - Replaces the -1 sentinel in Init_Win_bytes_* (+ flag for the backward direction)
    - Encodes Protocol as a single 0/1 column (Protocol_17 = UDP)
    - Applies One-Hot Encoding to Destination.Port (using SERVICE_PORTS + 'Other')
    """
    df = drop_duplicate_rows(df)

    # We preserve the groups for the split
    groups = df["Flow.ID"].copy()

    # We extract the target label
    y = df["ProtocolName"].copy()

    # We remove identifiers and target before calculating constants and duplicates
    # NOTE: L7Protocol contains the same info as ProtocolName
    id_cols = ["Flow.ID", "Source.IP", "Destination.IP", "Timestamp", "Source.Port", "ProtocolName", "L7Protocol"]
    X = df.drop(columns=[col for col in id_cols if col in df.columns]).copy()

    # We remove constant columns
    const_cols = get_constant_columns(X)
    X = X.drop(columns=const_cols)

    # We remove duplicate columns
    dup_cols = get_duplicate_columns(X)
    X = X.drop(columns=dup_cols)

    # We replace the -1 sentinel of the TCP initial window
    X = fix_init_win_sentinel(X)

    # Protocol has only two values (6=TCP, 17=UDP), so a single 0/1 column is enough
    X["Protocol_17"] = (X["Protocol"] == 17).astype(int)
    X = X.drop(columns=["Protocol"])

    # One-Hot Encoding for Destination.Port (known service ports + 'Other')
    X["Destination.Port"] = X["Destination.Port"].where(X["Destination.Port"].isin(SERVICE_PORTS), other="Other")
    X["Destination.Port"] = X["Destination.Port"].astype(str)

    X = pd.get_dummies(X, columns=["Destination.Port"], dtype=int)

    return X, y, groups


if __name__ == "__main__":
    processed_path = Path("data/processed/dataset_10apps.csv")
    if not processed_path.exists():
        raw_path = Path("data/raw/dataset.csv")
        if not raw_path.exists():
            print("Error: data/raw/dataset.csv not found. Please place it in data/raw/ first!")
            sys.exit(1)
        print("Saving the new csv....")
        df_raw = pd.read_csv("data/raw/dataset.csv", low_memory=False) #Low_memory=False to avoid warnings
        df_10apps = filter_10_apps(df_raw)
        df_10apps.to_csv(processed_path, index=False)
        dist = get_class_distribution(df_10apps)
        dist.to_csv("data/processed/class_distribution_10apps.csv")
        print("Csv file saved!")
    else:
        print(f"File {processed_path} already exists, skipping its creation")

    # Prepare dataset for the model
    print("\nPreparing dataset for model...")
    df_processed = load_dataset("data/processed/dataset_10apps.csv")
    X, y, groups = prepare_dataset_for_model(df_processed)

    # Save X with "label" (= y) column into model_ready.csv
    X_to_save = X.copy()
    X_to_save["label"] = y
    X_to_save.to_csv("data/processed/model_ready.csv", index=False)
    print("model_ready.csv saved successfully!")

    # Save groups ("Flow.ID" column) into groups.csv
    pd.DataFrame({"Flow.ID": groups}).to_csv("data/processed/groups.csv", index=False)
    print("groups.csv saved successfully!")

    # Informative prints
    print(f"X shape: {X.shape}")
    print(f"Number of distinct groups (for Flow.ID): {groups.nunique():,}")
    one_hot_cols = [c for c in X.columns if c.startswith("Protocol_") or c.startswith("Destination.Port_")]
    print(f"One-Hot columns created ({len(one_hot_cols)}): {one_hot_cols}")
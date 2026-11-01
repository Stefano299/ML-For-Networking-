from pathlib import Path
import pandas as pd
import numpy as np

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


def load_dataset(path="data/processed/dataset_10apps.csv"):
    return pd.read_csv(path, low_memory=False)


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


def make_ovr_target(y, app):
    if app not in APPS:
        raise ValueError(f"App {app} not valid, it must be in APPS")
    return (y == app).astype(int)


def prepare_dataset_for_model(df):
    """
    Prepares the dataset for model training:
    - Preserves groups = df["Flow.ID"].copy() for StratifiedGroupKFold
    - Removes identifiers: Flow.ID, Source.IP, Destination.IP, Timestamp, Source.Port
    - Separates target label (y = ProtocolName) and removes L7Protocol from features
    - Removes constant columns (only 1 unique value)
    - Removes duplicate columns (identical values)
    - Applies One-Hot Encoding to Protocol and Destination.Port (using SERVICE_PORTS + 'Other')

    Returns:
        (X, y, groups): tuple of DataFrames/Series aligned row-by-row
    """
    df = df.copy()

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

    # One-Hot Encoding for Protocol (6=TCP, 17=UDP)
    X["Protocol"] = X["Protocol"].astype(str)

    # One-Hot Encoding for Destination.Port (known service ports + 'Other')
    X["Destination.Port"] = X["Destination.Port"].where(X["Destination.Port"].isin(SERVICE_PORTS), other="Other")
    X["Destination.Port"] = X["Destination.Port"].astype(str)

    X = pd.get_dummies(X, columns=["Protocol", "Destination.Port"], dtype=int)

    return X, y, groups


if __name__ == "__main__":
    processed_path = Path("data/processed/dataset_10apps.csv")
    if not processed_path.exists():
        raw_path = Path("data/raw/dataset.csv")
        if not raw_path.exists():
            print("Error: data/raw/dataset.csv not found. Please place it in data/raw/ first!")
            exit(1)
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
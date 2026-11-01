"""
Lab #3 - Data Visualization & Network Traffic Analysis (Darknet Dataset)
Politecnico di Torino - Machine Learning for Networking / Data Science Lab

This file is formatted with '# %%' interactive cell markers for VS Code / Cursor / Jupyter.
It demonstrates the visualization toolchain (Matplotlib, Seaborn, Pandas) and network traffic
analysis patterns: timestamps, resampling, packet counts, network flows/links, and inter-arrival times.
"""

# %% [markdown]
# # Lab #3: Data Visualization and Network Traffic Characterization
#
# Objectives:
# 1. Visualization with Seaborn (`histplot`, `ecdfplot`, `boxplot`, `jointplot`, `lineplot`).
# 2. Pandas built-in plotting routines (`resample`, `plot`).
# 3. Practical Exercise on Network Traffic (`darknet_traces.csv`):
#    - Time-series resampling of packet rates (`pd.to_datetime`, `resample('60s').size()`).
#    - Flow/Link characterization: 4-tuple `(src_ip, src_port, dst_ip, dst_port)`.
#    - Packet inter-arrival times calculation via `.sort_values().diff()`.
#    - Log-scale boxplot analysis for network timing characteristics.

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% [markdown]
# ## 1. Tutorial: Visualization Techniques

# %% [markdown]
# ### 1.1 Seaborn Fundamentals
# Using the Iris dataset to showcase statistical plotting primitives.

# %%
from sklearn import datasets

iris_data = datasets.load_iris()
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
df_iris = pd.DataFrame(iris_data.data, columns=columns)
df_iris["species"] = "setosa"
df_iris.loc[50:99, "species"] = "versicolor"
df_iris.loc[100:149, "species"] = "virginica"

# Scatter plot with Seaborn
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df_iris, x="sepal_length", y="sepal_width", hue="species")
plt.title("Iris Sepal Dimensions by Species")
plt.show()

# Boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_iris, x="species", y="sepal_length")
plt.title("Boxplot of Sepal Length per Species")
plt.show()

# Distribution plot (Histogram + KDE)
plt.figure(figsize=(6, 4))
sns.histplot(data=df_iris, x="sepal_length", hue="species", kde=True, bins=15)
plt.title("Histogram with Kernel Density Estimation (KDE)")
plt.show()

# Empirical Cumulative Distribution Function (eCDF)
plt.figure(figsize=(6, 4))
sns.ecdfplot(data=df_iris, x="sepal_length", hue="species")
plt.title("Empirical Cumulative Distribution Function (eCDF)")
plt.grid(True, alpha=0.3)
plt.show()

# Joint plot (Scatter + Marginal KDEs)
sns.jointplot(data=df_iris, x="sepal_length", y="sepal_width", hue="species", kind="kde")
plt.show()

# %% [markdown]
# ---
# ## 2. Exercise: Darknet Network Traffic Analysis
#
# Dataset columns in `darknet_traces.csv`:
# - `ts`: Unix timestamp (seconds since Jan 1 1970)
# - `src_ip`, `src_port`: Source IP address and port
# - `dst_ip`, `dst_port`: Destination IP address and port
# - `proto`: Protocol used (TCP, UDP, ICMP...)
# - `pck_len`: Packet length in bytes
# - `ttl`: Time to Live counter

# %%
DATASET_PATH = Path("darknet_traces.csv")
if not DATASET_PATH.exists():
    DATASET_PATH = Path("../dataset.csv")  # fallback to parent dataset if present

if DATASET_PATH.exists():
    df_dark = pd.read_csv(DATASET_PATH, nrows=500000)  # limit rows for interactive speed if needed
    print("Darknet dataset loaded. Shape:", df_dark.shape)
    print(df_dark.head())
else:
    print("Dataset not found locally; generating synthetic darknet trace for illustration.")
    rng = np.random.default_rng(42)
    n = 10000
    df_dark = pd.DataFrame({
        "ts": 1570000000 + np.sort(rng.integers(0, 7200, n)),
        "src_ip": rng.choice([f"192.168.1.{i}" for i in range(1, 10)], n),
        "src_port": rng.integers(1024, 65535, n),
        "dst_ip": rng.choice(["10.0.0.1", "10.0.0.2"], n),
        "dst_port": rng.choice([80, 443, 8080, 22, 53], n),
        "proto": rng.choice([6, 17], n),
        "pck_len": rng.integers(40, 1500, n),
        "ttl": rng.integers(30, 128, n)
    })

# %% [markdown]
# ### 2.1 Plot the Number of Received Packets per Minute
# Workflow:
# 1. Convert `ts` to datetime with `pd.to_datetime(..., unit='s')`.
# 2. Set `ts` as the index.
# 3. Use `.resample('60s').size()` to aggregate packet rates.
# 4. Plot both time series and distribution (histplot + KDE).

# %%
df_dark_copy = df_dark.copy()
df_dark_copy["ts"] = pd.to_datetime(df_dark_copy["ts"], unit="s")
df_dark_copy.set_index("ts", inplace=True)

num_sec = 60
packets_per_interval = df_dark_copy.resample(f"{num_sec}s").size()

# Time-series line plot
plt.figure(figsize=(10, 4))
packets_per_interval.plot()
plt.xlabel("Timestamp")
plt.ylabel("Packet Count")
plt.title(f"Number of Packets per {num_sec} Seconds")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# Distribution plot with KDE
plt.figure(figsize=(8, 4))
sns.histplot(data=packets_per_interval, kde=True, bins=30)
plt.xlabel("Number of Packets")
plt.title(f"Distribution of Packets per {num_sec} Seconds")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.2 Packets per Minute for Top 5 Source IP Addresses

# %%
top_5_ips = df_dark["src_ip"].value_counts().head(5).index
print("Top 5 Source IPs:", list(top_5_ips))

df_top5 = df_dark_copy[df_dark_copy["src_ip"].isin(top_5_ips)]

# Resample per IP address
df_top5_resampled = (
    df_top5.groupby(["src_ip"])
    .resample(f"{num_sec}s")
    .size()
    .reset_index(name="packets")
)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1: Line plot over time
sns.lineplot(data=df_top5_resampled, x="ts", y="packets", hue="src_ip", ax=axs[0])
axs[0].set_title("Packet Rate over Time (Top 5 IPs)")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Packets / min")
axs[0].grid(True, alpha=0.4)

# Subplot 2: Histogram with KDE
sns.histplot(data=df_top5_resampled, x="packets", hue="src_ip", kde=True, ax=axs[1], element="step")
axs[1].set_title("Packet Rate Distribution")
axs[1].grid(True, alpha=0.4)

# Subplot 3: ECDF
sns.ecdfplot(data=df_top5_resampled, x="packets", hue="src_ip", ax=axs[2])
axs[2].set_title("ECDF of Packet Rate")
axs[2].grid(True, alpha=0.4)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.3 Study the Frequency of Specific Links
# A link (flow) is defined as `(src_ip, src_port, dst_ip, dst_port)`.
# We study:
# 1. Top 10 links with highest packet volume.
# 2. Scatter plot of packet arrivals over time per link.
# 3. Inter-arrival time distribution using `.diff()` on sorted timestamps.

# %%
# Group by link 4-tuple
links = df_dark.groupby(["src_ip", "src_port", "dst_ip", "dst_port"])
link_top_10 = links.size().nlargest(10).index

# %% [markdown]
# #### Scatter Plot of Packet Arrivals per Link

# %%
plt.figure(figsize=(12, 8))
ytick_labels = []

for n, link in enumerate(link_top_10):
    src_ip, src_port, dst_ip, dst_port = link
    df_link = links.get_group(link)
    
    # Scatter points along y = (10 - n)
    plt.scatter(
        pd.to_datetime(df_link["ts"], unit="s"),
        [10 - n for _ in range(len(df_link))],
        s=20,
        marker="*"
    )
    ytick_labels.append(f"{src_ip}:{src_port} -> {dst_ip}:{dst_port} ({len(df_link)} pkts)")

plt.xlabel("Timestamp")
plt.ylabel("Network Link")
plt.yticks(range(10, 0, -1), ytick_labels)
plt.title("Packet Transmissions for Top 10 Links")
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Inter-arrival Time Distribution (Boxplot with Log Scale)
# Key pattern:
# `intertime = df_link['ts'].sort_values().diff()[1:].values`
# Filter zeros: `intertime[intertime != 0]`
# Set logarithmic y-axis: `plt.yscale('log')`

# %%
plt.figure(figsize=(14, 6))
plt.title("Boxplot of Inter-arrival Times for Top 10 Links")
xlabels = []

for n, link in enumerate(link_top_10):
    src_ip, src_port, dst_ip, dst_port = link
    df_link = links.get_group(link)
    
    # Chronological sort then diff() gives inter-packet delay
    intertime = df_link["ts"].sort_values().diff()[1:].values
    intertime = intertime[intertime != 0]  # remove zero-latency bursts
    
    if len(intertime) > 0:
        plt.boxplot(intertime, positions=[n], widths=0.6)
    xlabels.append(f"{src_ip}:{src_port}\n-> {dst_port}\n({len(df_link)} pkts)")

plt.xlabel("Link")
plt.ylabel("Intertime [s] (log scale)")
plt.yscale("log")
plt.xticks(range(10), xlabels, rotation=45, ha="right")
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.show()

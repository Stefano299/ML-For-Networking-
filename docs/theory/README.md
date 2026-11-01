# ML4N Theory Knowledge Base

> **Course:** Machine Learning for Networking (ML4N)  
> **Institution:** Politecnico di Torino  
> **Course Instructor:** Prof. Luca Vassio  
> **Scope:** Lectures 2, 3, and 4 (Data Exploration, Preprocessing, and Dimensionality Reduction)

---

## Directory Overview

This directory contains the comprehensive theoretical foundations taught in the course. Each module provides rigorous mathematical definitions, slide cross-references (`[Slide XX]`), visualization guidelines, practical networking connections, and step-by-step solutions to official exam quizzes.

### Module Files

| Lecture | Document | Key Topics Covered |
| :---: | :--- | :--- |
| **Lecture 2** | [`02_Data_Exploration_and_Visualization.md`](02_Data_Exploration_and_Visualization.md) | Sample vs. Population statistics, KDE bandwidth, ECDF, Boxplot & $1.5 \times \text{IQR}$ rule, Outlier detection (GESD, DBSCAN), Pearson vs. Spearman correlation, Exam Quiz 1 |
| **Lecture 3** | [`03_Data_Preprocessing.md`](03_Data_Preprocessing.md) | Attribute scales (Nominal, Ordinal, Interval, Ratio), Distance metrics (Minkowski, Mahalanobis, Cosine, Jaccard), Min-Max vs. Z-Score normalization, Categorical encoding (One-Hot, Dummy, Effect), Text preprocessing (BoW, TF-IDF), Exam Quiz 2 |
| **Lecture 4** | [`04_Dimensionality_Reduction.md`](04_Dimensionality_Reduction.md) | Curse of dimensionality, Covariance matrix $\mathbf{S}$, Spectral decomposition, PCA derivation & proof of variance maximization, Scree plots & stopping rules, Loading scores, PCA vs. t-SNE, Exam Quiz 3 |
| **Consolidated** | [`ML4N_Theory_Full_Context.md`](ML4N_Theory_Full_Context.md) | **Single-file compilation** of Lectures 2, 3, and 4. Optimized for single-prompt LLM ingestion (Claude Projects, Cursor Context, or Workbench) |

---

## How an AI Assistant Should Use These Files

1. **Targeted Inquiries:** When working on a specific task (e.g., Min-Max normalization for adversarial attacks or PCA scree plots), consult the specific modular markdown file (`02_`, `03_`, or `04_`).
2. **Global Context Ingestion:** If feeding the entire theoretical course material into an LLM context window at once, use [`ML4N_Theory_Full_Context.md`](ML4N_Theory_Full_Context.md) to avoid multi-file fragmented loading.
3. **Mathematical Precision:** Retain all LaTeX formatting (`$...$` and `$$...$$`) and standard course notation:
   - $m$: number of instances (rows).
   - $n$: number of features (columns).
   - $p$: target reduced dimensionality ($p < n$).
   - $\mathbf{X}$: data matrix ($m \times n$).
   - $\mathbf{S}$: sample covariance matrix ($n \times n$).

# ML4N — Part 2: Data Exploration and Visualization
**Course:** Machine Learning for Networking (ML4N)  
**Instructor:** Prof. Luca Vassio (Politecnico di Torino)  
**Source Slides:** `2_ML4N_DataExplorationVisualization.pdf` (74 slides)  
**Purpose:** Optimized knowledge context for Claude / LLMs (noise-free, complete LaTeX formulas, consolidated slide animations, visual diagram descriptions, and worked quiz solutions).

---

## Table of Contents
1. [Core Notation & Mathematical Preliminaries](#1-core-notation--mathematical-preliminaries)
2. [Dataset Representation & Dimensionality](#2-dataset-representation--dimensionality)
3. [Univariate Analysis & Probability Distributions](#3-univariate-analysis--probability-distributions)
4. [Kernel Density Estimation (KDE)](#4-kernel-density-estimation-kde)
5. [Empirical Cumulative Distribution Function (ECDF)](#5-empirical-cumulative-distribution-function-ecdf)
6. [Summary Statistics, Percentiles & Boxplots](#6-summary-statistics-percentiles--boxplots)
7. [Outlier Detection Techniques](#7-outlier-detection-techniques)
8. [Multivariate Analysis & Correlation](#8-multivariate-analysis--correlation)
9. [Advanced Visualization Techniques](#9-advanced-visualization-techniques)
10. [Key Takeaways & Best Practices](#10-key-takeaways--best-practices)
11. [Self-Assessment Quiz (With Full Solutions)](#11-self-assessment-quiz-with-full-solutions)
12. [Readings & Acknowledgments](#12-readings--acknowledgments)

---

## 1. Core Notation & Mathematical Preliminaries
> 📍 *Reference: Slides 2–6*

### High-Level Framing
Machine Learning fits mathematical models to observed data in order to make predictions or forecasts. Data exploration and visualization is the critical first step to understand distribution shapes, anomalies, scale disparities, and feature correlations.

### Set and Vector Notation
| Symbol | Mathematical Meaning | Notes / Context |
| :--- | :--- | :--- |
| $a \in \mathcal{A}$ | Element membership | Object $a$ is an element of set $\mathcal{A}$ |
| $|\mathcal{A}|$ | Cardinality | Number of elements in a finite set $\mathcal{A}$ |
| $\mathcal{A} \subseteq \mathcal{B}$ | Subset | Every element in $\mathcal{A}$ belongs to $\mathcal{B}$ |
| $\mathcal{A} \subset \mathcal{B}$ | Strict subset | $\mathcal{A} \subseteq \mathcal{B}$ and $\mathcal{A} \neq \mathcal{B}$ |
| $\mathbb{N}$ | Natural numbers | $\mathbb{N} = \{1, 2, 3, \dots\}$ |
| $\mathbb{R}$ | Real numbers | Continuous 1D number line |
| $\{0, 1\}$ | Binary set | Discrete domain consisting only of values 0 and 1 |
| $[0, 1]$ | Unit interval | Closed continuous real interval $\{x \in \mathbb{R} \mid 0 \le x \le 1\}$ |
| $\mathbf{x} = (x_1, \dots, x_n)^T$ | Column vector | Vector in $\mathbb{R}^n$; $j$-th entry is denoted $x_j$ |
| $\|\mathbf{x}\|$ or $\|\mathbf{x}\|_2$ | Euclidean ($L_2$) Norm | Length of vector: $\|\mathbf{x}\|_2 = \sqrt{\sum_{j=1}^n x_j^2}$ |

---

## 2. Dataset Representation & Dimensionality
> 📍 *Reference: Slides 8–10*

### Tabular Data Formalism
A dataset $\mathbf{X}$ is typically represented as an $m \times n$ matrix:
- **Rows ($m$):** Data objects, records, points, samples, entities, or instances.
- **Columns ($n$):** Attributes, features, variables, or dimensions.

$$\mathbf{X} = \begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1n} \\ x_{21} & x_{22} & \cdots & x_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ x_{m1} & x_{m2} & \cdots & x_{mn} \end{bmatrix}$$

### Impact of Dataset Scale
1. **Computational Bottleneck:** Algorithms with high computational complexity ($O(m^2)$, $O(m^3)$, or $O(n^3)$) fail or exhaust RAM when $m$ or $n$ is very large.
2. **Mitigation Strategies:**
   - **Data Sampling:** Reduces the row dimension ($m' < m$) while preserving representative population distributions.
   - **Feature Selection / Dimensionality Reduction:** Reduces the column dimension ($n' < n$) by pruning redundant or irrelevant attributes.

### Attribute Properties
For every attribute $j \in \{1, \dots, n\}$, data exploration requires verifying:
- **Unit of Measurement:** e.g., seconds, bytes, packets/s, ms.
- **Attribute Type:** Categorical (Nominal, Ordinal) vs. Numerical (Discrete, Continuous).
- **Domain Constraints:** Ensuring values respect theoretical and physical bounds (e.g., latency $\ge 0$, probability $\in [0, 1]$).
- **Analysis Scope:** Univariate (individual feature properties) vs. Multivariate (inter-feature relations).

---

## 3. Univariate Analysis & Probability Distributions
> 📍 *Reference: Slides 11–24*

### Theoretical vs. Empirical Distributions
- **Theoretical Distribution:** Mathematical model of a random variable $X$, defined by parameters (e.g., mean $\mu$, variance $\sigma^2$, skewness, kurtosis).
  - **Discrete:** Characterized by a **Probability Mass Function (PMF)**:
    $$P(X = x_k) = p_k, \quad \sum_k p_k = 1$$
  - **Continuous:** Characterized by a **Probability Density Function (PDF)** $f(x)$:
    $$P(a \le X \le b) = \int_a^b f(x) \, dx, \quad \int_{-\infty}^\infty f(x) \, dx = 1$$
  - **Cumulative Distribution Function (CDF):**
    $$F(x) = P(X \le x) = \int_{-\infty}^x f(t) \, dt$$
    By the Fundamental Theorem of Calculus:
    $$f(x) = \frac{d}{dx} F(x)$$
- **Empirical Distribution:** Constructed from $m$ observed sample points $\{x_1, x_2, \dots, x_m\}$ drawn from an unknown underlying theoretical distribution.
  - Sample Mean: $\bar{x} = \frac{1}{m} \sum_{i=1}^m x_i$
  - Sample Variance / Std: $s^2 = \frac{1}{m-1} \sum_{i=1}^m (x_i - \bar{x})^2$, $s = \sqrt{s^2}$
  - Shape metrics: Sample Skewness (asymmetry) and Sample Kurtosis (tailedness / extreme event propensity).

### Histograms & Binning
- Divides the range $[\min(x), \max(x)]$ into contiguous, non-overlapping intervals (bins).
- Counts occurrences (frequencies) within each bin.
- **Trade-offs:**
  - Bin width too large $\to$ Oversmoothing, loss of distribution modality (e.g., merging bimodal peaks).
  - Bin width too small $\to$ Undersmoothing, noisy spikes driven by random sampling fluctuations.
  - *Data Artifacts in Real Histograms:* Non-uniformity due to human rounding (e.g., birth weights or survey responses where integers or multiples of 5/10 are disproportionately frequent).

### PMF Limitations & Difference of Distributions
- Visualizing raw PMFs with many distinct discrete values or multiple groups becomes visually cluttered and difficult to compare.
- **Remedies:**
  - Plotting the **Difference of PMFs** ($\Delta p(x) = p_1(x) - p_2(x)$) in percentage points across bins to highlight relative shifts.
  - Using the **Cumulative Distribution Function (CDF)**.

---

## 4. Kernel Density Estimation (KDE)
> 📍 *Reference: Slides 25–35*

### Purpose
Kernel Density Estimation is a non-parametric method to estimate the continuous Probability Density Function $f(x)$ directly from $m$ discrete samples without assuming a strict parametric model (like pure Gaussian).

### Mathematical Formulation
The estimated density $\widehat{f}_h(x)$ at any point $x$ is the weighted average of kernel functions centered at each data sample $x_i$:

$$\widehat{f}_h(x) = \frac{1}{m} \sum_{i=1}^m K_h(x - x_i) = \frac{1}{mh} \sum_{i=1}^m K\left(\frac{x - x_i}{h}\right)$$

Where:
- $m$: Number of observed samples.
- $h > 0$: **Bandwidth** (smoothing parameter).
- $K(z)$: **Kernel function**, satisfying non-negativity $K(z) \ge 0$ and unit area $\int_{-\infty}^\infty K(z) \, dz = 1$.

### Gaussian Kernel
The most standard choice is the standard normal density $\mathcal{N}(0, 1)$:

$$K(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}$$

Substituting into the KDE equation:

$$\widehat{f}_h(x) = \frac{1}{m h \sqrt{2\pi}} \sum_{i=1}^m \exp\left( -\frac{(x - x_i)^2}{2 h^2} \right)$$

### Bandwidth Selection ($h$) & Visual Behavior
- **Under-smoothed ($h$ too small):** Individual kernels do not overlap sufficiently; curve is spiky and captures sampling noise as false local modes.
- **Over-smoothed ($h$ too large):** Kernels spread excessively; flattens the distribution and erases true multimodality or distinct structural features.
- **KDE vs. Histogram:**
  - KDE provides a continuous, differentiable, less cluttered curve.
  - *Caveat:* KDE can suffer from edge boundary bias if the true distribution has hard physical bounds (e.g., $x \ge 0$).

---

## 5. Empirical Cumulative Distribution Function (ECDF)
> 📍 *Reference: Slides 36–37*

### Definition
The ECDF $\widehat{F}(x)$ is an unbiased step function estimator of the true CDF:

$$\widehat{F}(x) = \frac{1}{m} \sum_{i=1}^m \mathbf{1}_{\{x_i \le x\}}$$

Where $\mathbf{1}_{\{x_i \le x\}}$ is the indicator function (1 if $x_i \le x$, else 0).

### Key Properties
- Starts at 0 for $x < \min(x_i)$ and steps up by exactly $1/m$ at each observed data point, reaching 1 at $\max(x_i)$.
- **Parameter-free:** Unlike histograms and KDE, ECDF requires **no binning or bandwidth parameter**, avoiding arbitrary visual distortion.
- **Visual Interpretation:**
  - **Steep / vertical sections:** High density of observations (modes).
  - **Flat / horizontal sections:** Low density or absence of data points.

---

## 6. Summary Statistics, Percentiles & Boxplots
> 📍 *Reference: Slides 38–40*

### Percentiles & Quartiles
- **$p$-th Percentile:** The value below which $p\%$ of observations fall.
- **Quartiles:**
  - $Q_1$ (25th percentile): Lower quartile.
  - $Q_2$ (50th percentile): Median ($\tilde{x}$).
  - $Q_3$ (75th percentile): Upper quartile.
- **Interquartile Range (IQR):**
  $$\text{IQR} = Q_3 - Q_1$$

### Anatomy of a Boxplot (Tukey Boxplot)
- **Central Box:** Spans from $Q_1$ to $Q_3$ (contains the middle 50% of the data).
- **Median Line:** Horizontal/vertical line inside the box at $Q_2$.
- **Whiskers:** Extend from $Q_1$ and $Q_3$ to the furthest data points within:
  - Lower bound: $Q_1 - 1.5 \times \text{IQR}$
  - Upper bound: $Q_3 + 1.5 \times \text{IQR}$
- **Outliers:** Any individual sample point falling strictly beyond the whisker bounds, plotted as distinct markers (dots).

### Correspondence with Gaussian / Normal Distribution
For normally distributed data $X \sim \mathcal{N}(\mu, \sigma^2)$:
- Median $= \mu = 0\sigma$
- $Q_1 \approx \mu - 0.6745\sigma$, $Q_3 \approx \mu + 0.6745\sigma$
- $\text{IQR} \approx 1.349\sigma$
- Whiskers span:
  $$[\mu - 2.698\sigma, \; \mu + 2.698\sigma]$$
- **Theoretical outlier probability under normality:**
  - The probability of falling within the whiskers is approximately **99.30%**.
  - Exactly **0.70%** (about 1 in 143 samples) are flagged as potential outliers by chance even when the data is purely Gaussian.

---

## 7. Outlier Detection Techniques
> 📍 *Reference: Slides 41–44*

### Nature of Outliers
Outliers represent extreme deviations caused by:
1. Measurement, recording, or sensor transmission errors.
2. Legitimate, critical rare events (e.g., cyberattacks, flash network traffic, system failures).
*Handling rule:* Treatment requires **domain knowledge**; naive deletion can remove the most important signals.

### Univariate Detection: GESD Test
The **Generalized Extreme Studentized Deviate (GESD)** test (generalized Grubbs test) detects up to an upper bound $k$ of outliers in univariate data assuming an underlying normal distribution.

**Test Statistic:**
$$R_r = \frac{\max_{i} |x_i - \bar{x}|}{s}$$
Where $\bar{x}$ and $s$ are the sample mean and sample standard deviation of the current dataset.

**Iterative Procedure:**
1. Compute $R_1$, identify the observation maximizing $|x_i - \bar{x}|$, and temporarily remove it.
2. Recompute $\bar{x}$, $s$, and $R_2$ on the remaining $m-1$ points.
3. Repeat up to $k$ iterations to obtain statistics $R_1, R_2, \dots, R_k$.
4. Compare each $R_r$ against critical value $\lambda_r$ computed from the Student's $t$-distribution. The number of outliers is the largest $r \le k$ such that $R_r > \lambda_r$.

### Multivariate Detection: DBSCAN
**Density-Based Spatial Clustering of Applications with Noise (DBSCAN):**
- Non-parametric, clustering-based outlier detector in $\mathbb{R}^n$.
- Uses two hyperparameters: $\varepsilon$ (neighborhood radius) and $\text{MinPts}$ (minimum points).
- Classifies points into Core points, Border points, and **Noise points (Outliers)**. Points in low-density regions that do not belong to any dense cluster are explicitly labeled as outliers (label $-1$).

---

## 8. Multivariate Analysis & Correlation
> 📍 *Reference: Slides 45–62*

### Visualizing Bivariate Relationships
- **Scatter Plot:** Visual inspection of relationship shape (linear, non-linear, monotonic, clustered).
- **Overplotting Mitigations:**
  - **Jittering:** Adding small uniform random noise to discrete/rounded coordinates to reveal overlapping point counts.
  - **Alpha Blending:** Adding transparency so darker regions reflect higher local point density.
  - **Hexbin Plot:** 2D histogram using hexagonal spatial bins colored by point frequency.
  - **Binned Percentile Plots:** Slicing variable $X$ into bins and plotting the 25th, 50th, and 75th percentiles of $Y$ per bin.

### Covariance & Pearson Correlation Coefficient
Given two variables $\mathbf{x}, \mathbf{y}$ across $m$ samples:
- **Sample Covariance:**
  $$s_{xy} = \text{cov}(\mathbf{x}, \mathbf{y}) = \frac{1}{m-1} \sum_{k=1}^m (x_k - \bar{x})(y_k - \bar{y})$$
  *Limitation:* Scale-dependent; magnitude depends on units of measurement.

- **Pearson Correlation Coefficient ($r$):**
  $$\text{corr}(\mathbf{x}, \mathbf{y}) = \frac{s_{xy}}{s_x s_y} = \frac{\sum_{k=1}^m (x_k - \bar{x})(y_k - \bar{y})}{\sqrt{\sum_{k=1}^m (x_k - \bar{x})^2} \sqrt{\sum_{k=1}^m (y_k - \bar{y})^2}}$$

**Properties of Pearson $r$:**
- Strictly bounded: $-1 \le r \le +1$.
- Measures **strictly linear** associations.
- Scale-invariant.
- **Pitfall:** Uncorrelated does not imply independent! A perfect non-linear symmetric relationship (e.g., $y = x^2$ for $x \in \{-3, -2, -1, 0, 1, 2, 3\}$) yields $r = 0$.

### Spearman's Rank Correlation ($r_s$)
Non-parametric correlation measuring **monotonic** relationships (whether variables increase/decrease together, regardless of linearity).

**Procedure:**
1. Convert raw scores $x_i, y_i$ into ranks $\mathrm{R}(x_i), \mathrm{R}(y_i) \in \{1, \dots, m\}$.
2. Compute the Pearson correlation on the ranks:
   $$r_s = \frac{\text{cov}(\mathrm{R}(x), \mathrm{R}(y))}{\sigma_{\mathrm{R}(x)} \sigma_{\mathrm{R}(y)}}$$
3. **Simplified Formula (when all $m$ ranks are distinct integers):**
   $$r_s = 1 - \frac{6 \sum_{i=1}^m d_i^2}{m(m^2 - 1)}, \quad d_i = \mathrm{R}(x_i) - \mathrm{R}(y_i)$$

**Key Comparison:**
- For any strictly increasing monotonic non-linear relation (e.g., $y = e^x$ or $y = x^3$ for $x > 0$), Spearman $r_s = 1.0$, whereas Pearson $r < 1.0$.

---

## 9. Advanced Visualization Techniques
> 📍 *Reference: Slides 63–70*

### Heatmaps
- 2D grid representing pairwise feature correlation matrices.
- Cell $(i, j)$ represents $\text{corr}(x_i, x_j)$; color saturation denotes correlation strength (e.g., red $= +1$, blue $= -1$). Matrix is symmetric along the diagonal.

### Word Clouds
- Visual representation of unstructured text / document terms.
- Font size and color are proportional to term frequency or TF-IDF importance.

### Dimensionality Reduction for 2D Projection
- **PCA (Principal Component Analysis):** Projects high-dimensional points onto top 2 orthogonal axes of maximal variance (e.g., Iris 4D $\to$ 2D).
- **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Non-linear manifold embedding for cluster separation.
  - *Warning:* Distances between clusters in t-SNE are not preserved metrics and should not be used as raw input features for standard ML regressors without caution.

---

## 10. Key Takeaways & Best Practices
> 📍 *Reference: Slide 70*
1. Always explore raw data visually before fitting any machine learning model.
2. Select visualization tools aligned with data types (nominal $\to$ bar/frequency, continuous $\to$ KDE/ECDF, bivariate $\to$ scatter/hexbin).
3. Check for severe outliers and evaluate whether they represent noise or system anomalies.
4. Verify whether relationships are linear (Pearson) or monotonic non-linear (Spearman) before selecting regression models.

---

## 11. Self-Assessment Quiz (With Full Solutions)
> 📍 *Reference: Slide 72*

### Problem Statement
Given 12 observed samples:
$$\mathbf{x} = [0, 1, 1, 0, 5, 4.5, 7, 10, 0, 5, 6, -1]$$
$$\mathbf{y} = [5, 1, 7, 4, 9, 8, 7, 15, 5, 9, 9, 5]$$

1. Compute descriptive statistics for $\mathbf{x}$ (Sorted, Mean, Std, Median, Quartiles, IQR, Whiskers).
2. Determine if $\mathbf{x}$ contains boxplot outliers.
3. Compute the Pearson correlation coefficient $r(\mathbf{x}, \mathbf{y})$.
4. Compute the Spearman rank correlation coefficient $r_s(\mathbf{x}, \mathbf{y})$.

---

### Step-by-Step Solutions

#### 1. Distribution & Boxplot Statistics for $\mathbf{x}$
- **Sorted array ($m=12$):**
  $$\mathbf{x}_{\text{sorted}} = [-1.0, \; 0.0, \; 0.0, \; 0.0, \; 1.0, \; 1.0, \; 4.5, \; 5.0, \; 5.0, \; 6.0, \; 7.0, \; 10.0]$$
- **Sample Mean ($\bar{x}$):**
  $$\bar{x} = \frac{-1 + 0 + 0 + 0 + 1 + 1 + 4.5 + 5 + 5 + 6 + 7 + 10}{12} = \frac{38.5}{12} \approx 3.2083$$
- **Sample Standard Deviation ($s_x$, $ddof=1$):**
  $$s_x = \sqrt{\frac{1}{11} \sum_{i=1}^{12} (x_i - \bar{x})^2} \approx 3.4997$$
- **Median ($Q_2$):**
  Average of 6th and 7th elements: $\frac{1.0 + 4.5}{2} = 2.75$
- **Quartiles & IQR:**
  - $Q_1$ (25th percentile) $= 0.0$
  - $Q_3$ (75th percentile) $= 5.25$
  - $\text{IQR} = Q_3 - Q_1 = 5.25 - 0.0 = 5.25$
- **Whisker Thresholds ($1.5 \times \text{IQR}$):**
  - Lower boundary: $Q_1 - 1.5 \times \text{IQR} = 0 - 1.5(5.25) = -7.875$
  - Upper boundary: $Q_3 + 1.5 \times \text{IQR} = 5.25 + 1.5(5.25) = 13.125$

#### 2. Outlier Assessment
Since all samples in $\mathbf{x}$ lie in $[-1.0, 10.0]$, every sample satisfies:
$$-7.875 \le x_i \le 13.125$$
**Conclusion:** There are **no boxplot outliers** in dataset $\mathbf{x}$.

#### 3. Pearson Correlation $r(\mathbf{x}, \mathbf{y})$
- Mean of $\mathbf{y}$: $\bar{y} = \frac{84}{12} = 7.000$
- Sum of squared deviations: $\sum (x_i - \bar{x})^2 = 134.729$, $\sum (y_i - \bar{y})^2 = 134.000$
- Covariance numerator: $\sum (x_i - \bar{x})(y_i - \bar{y}) = 112.500$
- Correlation:
  $$r = \frac{112.500}{\sqrt{134.729 \times 134.000}} = \frac{112.500}{134.364} \approx \mathbf{0.8373}$$
**Interpretation:** Strong positive linear correlation.

#### 4. Spearman Rank Correlation $r_s(\mathbf{x}, \mathbf{y})$
Computing average ranks for tied values:
- Ranks for $\mathbf{x}$: $[3.0, 5.5, 5.5, 3.0, 8.5, 7.0, 11.0, 12.0, 3.0, 8.5, 10.0, 1.0]$
- Ranks for $\mathbf{y}$: $[4.0, 1.0, 7.5, 2.0, 10.5, 9.0, 7.5, 12.0, 4.0, 10.5, 10.5, 4.0]$
- Spearman correlation:
  $$r_s \approx \mathbf{0.7882}$$
**Interpretation:** Strong positive monotonic relationship.

---

## 12. Readings & Acknowledgments
> 📍 *Reference: Slides 73–74*
- **Readings:**
  - Tan, Steinbach, Kumar, *Introduction to Data Mining*, McGraw Hill (2006) — Chapters 1 & 9.
  - Allen B. Downey, *Think Stats*, O'Reilly.
  - Alice Zheng, Amanda Casari, *Feature Engineering for Machine Learning*, O'Reilly.
- **Acknowledgments:** Prof. Tania Cerquitelli and Prof. Elena Maria Baralis (Politecnico di Torino).

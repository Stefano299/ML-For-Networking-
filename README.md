# ML4N — Project 4: Adversarial Attacks and Defenses on Network Traffic Classifiers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![ART](https://img.shields.io/badge/ART-1.15+-green.svg)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

> **Course:** Machine Learning for Networking (ML4N) / Data Science Lab  
> **Institution:** Politecnico di Torino  
> **Instructor:** Prof. Luca Vassio  
> **Tutor:** Franco Galante (`franco.galante@polito.it`)  
> **Task:** Group Project 4 — Tabular IP flow classification (10 target apps), adversarial evasion attacks (FGSM/PGD via ART), adversarial training, backdoor poisoning, and defender clustering.

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd "ML Lab"
```

### 2. Create and Activate Virtual Environment
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💾 Dataset Setup

> **Note on File Sizes:**  
> The raw dataset (`dataset.csv`) is **~1.7 GB**, and processed matrices exceed **200 MB**. They are strictly excluded by `.gitignore` to prevent exceeding GitHub's 100 MB file limit.

### Step 1: Place the Raw Dataset
Download `dataset.csv` from the official course source / Kaggle and place it in the `data/raw/` directory:
```text
data/
└── raw/
    ├── .gitkeep
    └── dataset.csv   <-- Place your downloaded raw CSV here
```

### Step 2: Run Data Preparation Pipeline
Execute the data processing script to extract the 10 target applications, compute class distributions, drop redundant/constant features, apply One-Hot Encoding on service ports, and generate model-ready datasets:

```bash
python -m src.data
```

This will automatically populate `data/processed/` with:
- `dataset_10apps.csv`: Flows filtered for the 10 target applications (~270 MB).
- `model_ready.csv`: Feature matrix $X$ aligned with `label` $y$ (~205 MB).
- `groups.csv`: `Flow.ID` identifiers used for `StratifiedGroupKFold` (~19 MB).
- `class_distribution_10apps.csv`: Summary table of flow counts and class percentages.

---

## 🗂️ Project Architecture

```text
.
├── .gitignore                     # Comprehensive git ignore rules (data, weights, temp files)
├── README.md                      # [THIS FILE] Project overview and setup guide
├── requirements.txt               # Pinned Python package dependencies
├── data/
│   ├── raw/                       # Raw input files (ignored by Git)
│   │   └── .gitkeep
│   └── processed/                 # Generated feature matrices and splits (ignored by Git)
│       └── .gitkeep
├── docs/                          # Comprehensive technical knowledge base
│   ├── README.md                  # Documentation navigation guide
│   ├── CODING_STYLE_GUIDE.md      # Professor's coding rules (vectorization, OOP ML, styling)
│   ├── project/
│   │   ├── PROJECT_SPECIFICATIONS.md    # Official task specs (Sections 1 to 4)
│   │   ├── PROJECT_WORKFLOW_GUIDE.md    # Step-by-step roadmap and pitfalls
│   │   ├── professor_clarifications.md  # Q&A / technical nuances for the instructor
│   │   └── project-instructions.pdf     # Official course assignment PDF
│   ├── theory/                    # Lecture notes (EDA, Preprocessing, PCA)
│   └── labs/                      # Reference implementations from course laboratories
├── notebooks/                     # Interactive Jupyter notebooks for EDA and experiments
│   ├── .gitkeep
│   └── 01_eda.ipynb               # Exploratory data analysis starter notebook
├── results/                       # Output artifacts (ignored by Git, tracked via .gitkeep)
│   ├── models/                    # Saved PyTorch checkpoints and trained estimators
│   └── plots/                     # Generated figures, confusion matrices, and ROC curves
└── src/                           # Reusable Python source package
    ├── __init__.py
    └── data.py                    # Dataset loading, filtering, and preprocessing pipeline
```

---

## 📖 Key Documentation Links

Before contributing code, please consult the guides inside [`docs/`](docs/):
- **[`docs/CODING_STYLE_GUIDE.md`](docs/CODING_STYLE_GUIDE.md)**: Rules enforced by the professor (NumPy vectorization, no Python loops over rows, `.to_numpy()`, Scikit-Learn OOP design, `np.random.default_rng`).
- **[`docs/project/PROJECT_SPECIFICATIONS.md`](docs/project/PROJECT_SPECIFICATIONS.md)**: Official project requirements (classifiers, ART gradient attacks, backdoor poisoning).
- **[`docs/project/PROJECT_WORKFLOW_GUIDE.md`](docs/project/PROJECT_WORKFLOW_GUIDE.md)**: Step-by-step implementation guide.
- **[`docs/project/professor_clarifications.md`](docs/project/professor_clarifications.md)**: Doubts and design choices regarding `Source.Port`, inverted flows, and TCP window sentinels.

---

## 🤝 Team Git Workflow & Best Practices

1. **Never commit large files**:
   Never force-add files inside `data/raw/` or `data/processed/`, or `.pt`/`.pth` weights larger than 50 MB.


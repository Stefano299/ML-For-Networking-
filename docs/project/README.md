# ML4N Project Directory

This directory contains all documentation and guidelines for **Group Project 4: Analyzing Adversarial Attacks on Network Data Classifiers**.

## Documents in this Directory

- [`PROJECT_SPECIFICATIONS.md`](PROJECT_SPECIFICATIONS.md): The official project requirements from the course instructors (Prof. Luca Vassio, Tutor Franco Galante). Defines the 4 project sections, mandatory constraints, dataset details, and evaluation criteria.
- [`PROJECT_WORKFLOW_GUIDE.md`](PROJECT_WORKFLOW_GUIDE.md): Operational guide detailing how to execute the project end-to-end, mapping theoretical concepts to code modules in `src/` and interactive notebooks in `notebooks/`.

## Project Mapping to Codebase

| Project Section | Planned Notebook | Target Module in `src/` |
| :--- | :--- | :--- |
| **Section 1: Data Acquisition, Filtering & EDA** | `notebooks/01_eda.ipynb` | `src/data.py` |
| **Section 2: Supervised Classifiers & PyTorch MLP** | `notebooks/02_models.ipynb` | `src/models.py`, `src/evaluation.py` |
| **Section 3: Adversarial Attacks (FGSM, PGD, ART)** | `notebooks/03_adversarial.ipynb` | `src/attacks.py`, `src/evaluation.py` |
| **Section 4: Backdoor Poisoning & Clustering Defense** | `notebooks/04_backdoor.ipynb` | `src/attacks.py` |

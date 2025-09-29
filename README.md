# MultiPhysio-HRC: Multimodal Physiological Signals Dataset for Industrial Human-Robot Collaboration

> Companion GitHub repository for the paper **“MultiPhysio-HRC: Multimodal Physiological Signals Dataset for Industrial Human-Robot Collaboration.”**  
> This repo hosts **code, preprocessing pipelines, feature extraction, and baseline models** to reproduce results from the paper.

<p align="center">
  <a href="#citation"><img alt="Cite this" src="https://img.shields.io/badge/Cite-this-blue"></a>
  <a href="#getting-started"><img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue"></a>
  <a href="#license"><img alt="License" src="https://img.shields.io/badge/Code-MIT-blue"></a>
  <a href="https://doi.org/XXXX"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-Zenodo-brightgreen"></a>
  <img alt="Status" src="https://img.shields.io/badge/Status-Active-success">
</p>

---

## 📖 Overview
**MultiPhysio-HRC** is a multimodal dataset and toolkit for **mental-state estimation** in **industrial Human-Robot Collaboration (HRC)**.  

This repository provides:
- Loaders and preprocessing for physiological, EEG, voice, and AU data.
- Feature extraction pipelines for all modalities.
- Baseline regression and classification models with LOSO-CV evaluation.
- Example notebooks to explore the dataset and reproduce paper results.

---

## 📂 Repository Structure

```
.
├─ paper/                          # Camera‑ready or preprint PDF, figures (optional)
├─ docs/
│  ├─ dataset_overview.md          # Modalities, tasks, questionnaires, ethics
│  ├─ data_schema.md               # File formats, splits, naming, timestamps
│  └─ benchmarks.md                # Baseline setups & expected metrics
├─ src/
│  ├─ dataprep/                    # Loading, syncing, cleaning
│  ├─ features/                    # Physio, EEG, voice, AUs feature extraction
│  ├─ models/                      # Baselines (RF/AB/XGB), utils
│  └─ eval/                        # Metrics, LOSO CV, reporting
├─ notebooks/
│  ├─ 01_quicklook.ipynb           # Explore a subject & modalities
│  ├─ 02_extract_features.ipynb    # End‑to‑end feature extraction
│  └─ 03_train_baselines.ipynb     # Reproduce results from the paper
├─ examples/
│  └─ minimal_pipeline.py          # Scripted end‑to‑end run
├─ requirements.txt                # Python deps
├─ pyproject.toml                  # (optional) for modern builds
├─ CITATION.cff                    # Paper metadata (fill in DOI when available)
└─ README.md                       # You are here
```

> **Tip:** If you keep raw data outside the repo, set `MULTIPHYSIO_HRC_DATA` env var to the dataset root to avoid passing paths around.

---

## Getting Started
### 1) Install
```bash
# create a clean env (conda or mamba recommended)
conda create -n mphrc python=3.10 -y
conda activate mphrc

# install dependencies
pip install -r requirements.txt
```

### 2) Download the dataset
- Visit **https://tinyurl.com/MultiPhysio-HRC** and follow the instructions to obtain access and download files.
- Keep the raw data in a folder of your choice and set:
```bash
export MULTIPHYSIO_HRC_DATA=/path/to/MultiPhysio-HRC
```

### 3) Sanity‑check a subject
```bash
jupyter lab  # then open notebooks/01_quicklook.ipynb
```

---

## Data Schema
```
MultiPhysio-HRC/
│
├── physiological_data/
│   ├── filtered/                # Preprocessed signals
│   │   ├── subj1/
│   │   │   ├── task1.csv
│   │   │   ├── task2.csv
│   │   │   ...
│   │   └── subj2/
│   │       ├── task1.csv
│   │       ├── task2.csv
│   │       ...
│   │
│   └── raw/                     # Raw signals as acquired
│       ├── subj1/
│       │   ├── task1.csv
│       │   ├── task2.csv
│       │   ...
│       └── subj2/
│           ├── task1.csv
│           ├── task2.csv
│           ...
│
├── features/                    # Extracted features and labels
│   ├── aus_data.csv
│   ├── bio_features_60s.csv
│   ├── eeg_features_5s.csv
│   ├── nlp_embeddings.csv
│   ├── speech_features.csv
│   └── labels.csv
|
└── README.md
```
- **Windows:** Physio features on 60 s windows; EEG on 5 s windows; AUs at 2 fps.

---

## Reproducing the Paper Baselines
**End‑to‑end (script):**
```bash
python examples/minimal_pipeline.py \
  --data $MULTIPHYSIO_HRC_DATA \
  --modality physio \
  --task regression --label STAI \
  --cv loso --report out/report_physio_stai.json
```

**Notebooks:**
1. `02_extract_features.ipynb` – computes features for Physio/EEG/Voice/AUs.
2. `03_train_baselines.ipynb` – trains RF / AdaBoost / XGBoost for regression & 3‑class classification (Low/Med/High) based on per‑subject z‑like thresholds.

**Models:** RandomForest, AdaBoost, XGBoost. Evaluation uses **Leave‑One‑Subject‑Out (LOSO)**. Features & labels are min–max normalized within‑subject as in the paper.

---

## Results (from the paper)
- **Regression (STAI‑Y1 & NASA‑TLX):** Physiological features yield the **lowest RMSE**, stronger than EEG and Voice.
- **3‑Class Classification (Stress & Cognitive Load):** Physiological features achieve the **highest F1**, with EEG close behind for cognitive load; Voice trails Physio/EEG.

> See `docs/benchmarks.md` for expected ranges and how we compute the Low/Med/High bins per subject.

---

## FAQ
**Q: How do I get access to raw videos or robot logs?**  
A: See the dataset page. Some assets may require additional request/agreements.

**Q: Are there ready‑made splits?**  
A: We default to **Leave‑One‑Subject‑Out**. Utility functions can generate stratified splits by task/condition.

---

## Citation
If you use **MultiPhysio‑HRC** or this code, please cite the paper:

```bibtex
@article{MultiPhysioHRC2025,
  title   = {MultiPhysio-HRC: Multimodal Physiological Signals Dataset for industrial Human-Robot Collaboration},
  author  = {Bussolan, Andrea and Baraldo, Stefano and Avram, Oliver and Urcola, Pablo and Montesano, Luis and Gambardella, Luca Maria and Valente, Anna},
  year    = {2025},
  journal = {TBD},
  volume  = {TBD}, number = {TBD}, pages = {TBD},
  doi     = {TBD},
}
```

---

## Acknowledgments & Funding
- **Horizon Europe — FLUENTLY** (Grant **101058680**)
- **Eurostars — !2309‑Singularity**
- We thank all participants and the technical staff who supported the acquisition campaign.

---

## Ethics & License

This dataset was collected under institutional ethical approval (SUPSI), with informed consent from all participants.
- **Code:** Licensed under the [MIT License](https://opensource.org/licenses/MIT).  
- **Dataset:** Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).  

---

## Contact
- Lead contact: **andrea.bussolan@supsi.ch**
- Issues & questions: please open a GitHub issue.

---

*Maintainers:* Andrea Bussolan, Stefano Baraldo.
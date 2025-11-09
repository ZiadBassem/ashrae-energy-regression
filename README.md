# 🏠 ASHRAE – Great Energy Predictor III  

## Overview
My first Kaggle competition and a full‑hourly energy‑consumption model built from the ground up.

This repository contains the complete workflow:
- Data wrangling & weather/building merges  
- Robust cleaning (clipping & median imputation)  
- Log‑target transformation (`np.log1p`)  
- Power (Yeo‑Johnson) transforms on scale features  
- Regularized LightGBM regressor with sub‑sampling and early stopping  
- Full‑hourly inference (41 M rows) on 16 GB RAM hardware  
- Feature importance and interpretation plots  

---

## Key Files

| File | Description |
|------|--------------|
| `notebooks/04_kaggle_final_submission.ipynb` | complete training + submission workflow |
| `outputs/submission_refined_lgb.csv` | Kaggle submission (41 M rows × 2 columns) |
| `outputs/lightgbm_full_refined.pkl` | serialized LightGBM model |
| `outputs/run_log.json` | training metadata (RMSE, params, timestamp) |
| `requirements.txt` | environment specifications |

---

## Environment & Setup

```bash
git clone https://github.com/<your_username>/ashrae-energy-regression.git
cd ashrae-energy-regression
conda create -n ashrae python=3.10
conda activate ashrae
pip install -r requirements.txt
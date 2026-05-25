# Solar Radiation Forecasting — ML/DL Hybrid Ensemble

A research pipeline for predicting **all-sky solar radiation** using hybrid deep learning + machine learning ensemble models, with Optuna hyperparameter tuning, SHAP explainability, and statistical significance testing.

---

## 📁 Project Structure

```
Solar_Forecasting/
├── model_final.ipynb          # Main notebook (all 8 blocks)
├── reduced_dataset.csv        # Input dataset (required)
├── analysis.csv  # Pre-Processing
└── README.md
```

---

## ⚙️ Environment Setup

### 1. Create Virtual Environment

```bash
python -m venv solar_env
```

### 2. Activate It

```bash
# macOS / Linux
source solar_env/bin/activate

# Windows
solar_env\Scripts\activate
```

### 3. Install All Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost \
            tensorflow scikeras keras-tcn optuna shap \
            matplotlib seaborn scipy tqdm jupyter
```

> **Note:** If you're on an Apple Silicon Mac (M1/M2/M3), replace `tensorflow` with `tensorflow-macos` and `tensorflow-metal` for GPU acceleration.

### 4. Launch Jupyter

```bash
jupyter notebook model_final.ipynb
```

---

## 📦 Key Libraries

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading and manipulation |
| `scikit-learn` | Train/test split, scaling, stacking, metrics |
| `xgboost` | XGBoost regressor |
| `lightgbm` | LightGBM regressor |
| `catboost` | CatBoost regressor |
| `tensorflow` / `keras` | BiLSTM, BiGRU deep learning models |
| `scikeras` | Wraps Keras models for sklearn compatibility |
| `keras-tcn` | Temporal Convolutional Network (TCN) |
| `optuna` | Bayesian hyperparameter optimisation |
| `shap` | Model explainability (SHAP values) |
| `scipy` | Paired t-test, Wilcoxon signed-rank test |
| `matplotlib`, `seaborn` | All visualisations |
| `tqdm` | Progress bars |

---

## 📊 Dataset


The original raw dataset is the NASA Dataset for Riyadh, published on IEEE DataPort by Farrukh Hafeez (DOI: 10.21227/mrmj-vp45). It contains meteorological variables related to solar power generation — including solar irradiance, air temperature, wind speed, and humidity for Riyadh, Saudi Arabia — collected via NASA satellite imagery and ground stations, with the goal of improving solar radiation forecasting models under varying climatic conditions. The file approah_1_fe_datset.csv (also hosted in the same IEEE DataPort repository) is a feature-engineered version of that raw data, extended to 33 columns across 24,913 records by adding lag features, rolling statistics, interaction terms, a Fourier transform of solar radiation, and datetime components. The file reduced_dataset.csv is a further feature-selected version of that engineered dataset, retaining only 25 columns — dropping Datetime, raw Precipitation, Temp_Humidity_Interaction, Precipitation_Rolling_Mean, and all four rolling standard deviation features — while keeping row count and all other columns identical. Note that Month and Day of Week are typed as float64 in the reduced file rather than int64 as in the engineered version. Neither file contains null values. ieee-dataport

**File required:** `reduced_dataset.csv`

- **Shape:** 24,913 rows × 25 columns
- **Target:** `Solar Radiation (All Sky)` — solar irradiance in W/m²
- **Features include:** Air Temperature, Wet Bulb Temperature, Humidity, Wind Speed, Dew Point, and their lag/rolling features, plus Fourier transform and clear-sky radiation

### ⚠️ Leakage Columns Removed Before Training

The following columns are highly correlated with the target and removed to prevent data leakage:

```
Solar_Radiation_(Clear_Sky)          # 0.99 correlation — direct leakage
Solar_Radiation_(All_Sky)_Lag_1      # 0.95 correlation
Solar_Radiation_(All_Sky)_Lag_2
Solar_Radiation_(All_Sky)_Lag_3
Fourier_Transform_Solar_Radiation
```

After removal: **19 clean features** remain.

---

## 🧠 What the Code Does — Block by Block

### Block 1 — Baseline Models (Unoptimised)
Trains 5 models (RandomForest, XGBoost, LGBM, CatBoost, RF+ExtraTrees ensemble) and evaluates R², RMSE, MAE, MAPE on a random 80/20 split.

---

### Block 2 — Leakage Detection & Validation
Runs correlation analysis, duplicate checks, and compares **random split vs time-based split** R² to detect temporal or feature leakage. Also checks feature importance to confirm no single feature dominates suspiciously.

---

### Block 3 — Clean Hybrid Run (BiLSTM + BiGRU → Boosting)
**Config 1:** Trains BiLSTM and BiGRU as base learners → appends their predictions as extra features → trains XGBoost / LGBM / CatBoost as meta-learners.

**Config 2:** Trains XGBoost + LGBM + CatBoost as base learners → stacks them under RandomForest / ExtraTrees meta-learners.

Results over **3 repeated runs** with mean ± std reported.

---

### Block 4 — Optuna Hyperparameter Tuning
Uses **Optuna TPE sampler** with 30 trials each to tune:
- XGBoost: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- LightGBM: same + `num_leaves`
- CatBoost: `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`
- RandomForest & ExtraTrees: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`

---

### Block 5 — Full Comparison: Unoptimised vs Optimised (3 Runs)
Runs both versions across 3 random seeds and produces a clean **mean ± std table** for all base models and meta-learners across both configs.

---

### Block 6 — TCN Integration + Full Rerun

**New base model added: TCN (Temporal Convolutional Network)**

Steps:
1. TCN trained unoptimised (standalone base)
2. TCN tuned via Optuna (30 trials) — best: `nb_filters=128`, `kernel_size=3`, `dilations=[1,2,4,8]`
3. TCN optimised standalone base evaluated
4. **Config 1 rerun:** BiLSTM + BiGRU + **TCN** → XGB / LGBM / CatBoost
5. **Config 2 rerun:** BiLSTM + BiGRU + **TCN** + XGB + LGBM + CatBoost → RF / ET (manual stacking surface, not sklearn StackingRegressor)

Final unified ranking of all meta-learners (optimised):

| Rank | Model | Config | R² |
|---|---|---|---|
| 1 | Mix_ET | Config2 (with TCN) | 0.9624 |
| 2 | Mix_ET | Config2 (TCN version) | 0.9610 |
| 3 | Mix_RF | Config2 (with TCN) | 0.9609 |
| 4 | DL_CatBoost | Config1 | 0.9605 |
| 5 | Mix_RF | Config2 (TCN version) | 0.9568 |

---

### Block 7 — SHAP Explainability

Computes **SHAP values** for all 5 optimised meta-learners.

**Important:** DL prediction columns (`bilstm`, `bigru`, `tcn`) and ML base columns (`xgb_base`, `lgbm_base`, `cat_base`) are **used in training but excluded from SHAP plots** so only original meteorological features are visualised.

Outputs per model:
- `shap_*_bar.png` — Mean |SHAP| horizontal bar chart
- `shap_*_heatmap.png` — Top 15 features × 10 sample groups heatmap
- `shap_*_table.csv` — Ranked feature importance table

**Top features across models:** `Air_Temperature_Lag_3`, `Temp_Rolling_Mean`, `Month`, `Wind_Speed_Rolling_Mean`, `Air_Temperature`

Combined 5-model comparison also saved as `shap_comparison_all5_bar.png` and `shap_comparison_all5_heatmap.png`.

---

### Block 8 — Statistical Significance Testing

**Paired t-test (Optimised vs Unoptimised)** across all 5 meta-learners and 4 metrics.

Key findings:

| Model | R² Significance | RMSE Significance |
|---|---|---|
| DL_CatBoost | *** (p=0.00092) | *** (p=0.00029) |
| Mix_RF | *** (p=0.00040) | *** (p=0.00027) |
| Mix_ET | ** (p=0.00842) | ** (p=0.00823) |
| DL_XGB | ** (p=0.00704) | ** (p=0.00753) |
| DL_LGBM | * (p=0.02801) | * (p=0.02666) |

All 5 models show **statistically significant improvement** with Optuna tuning across all metrics.

Outputs: `graph_config1_comparison.png`, `graph_config2_comparison.png`, `ttest_significance_summary.csv`

---

## 🔑 Key Results Summary

| Model | Config | R² (Optimised) | RMSE | MAE | MAPE |
|---|---|---|---|---|---|
| **Mix_ET** | Config2 + TCN | **0.9624** | 0.0322 | 0.0149 | 2.15% |
| Mix_RF | Config2 + TCN | 0.9609 | 0.0328 | 0.0153 | 2.22% |
| DL_CatBoost | Config1 | 0.9605 | 0.0334 | 0.0167 | 2.39% |
| DL_XGB | Config1 | 0.9407 | 0.0405 | 0.0192 | 2.68% |
| DL_LGBM | Config1 | 0.9389 | 0.0411 | 0.0205 | 2.89% |

---

## 🚀 How to Run

Run the notebook blocks **in order** — each block depends on variables from the previous one (especially `best_xgb_params`, `best_lgbm_params`, etc. from Block 4 being in memory for Blocks 5–8).

```
Block 1 → Block 2 → Block 3 → Block 4 → Block 5 → Block 6 → Block 7 → Block 8
```

If you restart the kernel, re-run from Block 4 onwards before running Blocks 5–8.

---

## ⚠️ Common Issues

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: tcn` | `pip install keras-tcn` |
| `ModuleNotFoundError: scikeras` | `pip install scikeras` |
| `best_xgb_params not defined` | Re-run Block 4 before Block 5/6/7/8 |
| CatBoost `n_estimators` vs `iterations` key error | Already handled in code; ensure you run the param-fix cell |
| Memory error on large SHAP computation | Reduce `X_test_h` sample size or use `shap.sample` background |

---


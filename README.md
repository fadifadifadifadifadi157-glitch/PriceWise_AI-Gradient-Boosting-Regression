# 🏡 House Price Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

**An end-to-end Machine Learning web app** that predicts house sale prices using a hyperparameter-tuned Gradient Boosting model trained on the Ames Housing dataset.

[✨ Live Demo](#-live-demo) • [🚀 Quick Start](#-quick-start) • [📐 Architecture](#-project-architecture) • [📊 Model Results](#-model-results)

</div>

---

## 📸 Preview

> Premium dark glassmorphism UI with 7 organized input tabs, interactive feature importance charts, and instant ML-powered price predictions.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **ML-Powered** | Gradient Boosting Regressor with GridSearchCV hyperparameter tuning |
| 🎨 **Premium UI** | Dark glassmorphism design with Poppins font, animated buttons & gradient cards |
| 📋 **7 Input Tabs** | Location, Construction, Basement, Interior, Garage, Outdoor, Sale Info |
| 🔥 **Feature Importance** | Interactive Plotly chart showing the top 15 model drivers — always visible in sidebar |
| ⚡ **Quick Fill** | 3 sample profiles (Starter / Family / Luxury) to demo predictions instantly |
| 📊 **Model Metrics** | Test R² ≈ 0.898, CV R² ≈ 0.868, MAE ≈ $17,400 displayed in the sidebar |
| ⬇️ **CSV Export** | Download your input features + predicted price as a `.csv` file |
| 🔁 **Caching** | `@st.cache_resource` model loading + `@st.cache_data` importance computation |

---

## 📐 Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    house_price_analysis.ipynb               │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────────────┐ │
│  │  Data Load   │→ │  EDA &     │→ │  Model Training     │ │
│  │  train.csv   │  │  Cleaning  │  │  (6 algorithms)     │ │
│  └──────────────┘  └────────────┘  └──────────┬──────────┘ │
│                                               │             │
│                                    ┌──────────▼──────────┐  │
│                                    │  GridSearchCV +     │  │
│                                    │  5-Fold CV Tuning   │  │
│                                    └──────────┬──────────┘  │
│                                               │             │
│                                    ┌──────────▼──────────┐  │
│                                    │  gradient_boosting  │  │
│                                    │  _model.pkl         │  │
│                                    └──────────┬──────────┘  │
└───────────────────────────────────────────────┼─────────────┘
                                                │  joblib.load
┌───────────────────────────────────────────────▼─────────────┐
│                         app.py (Streamlit)                  │
│                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────┐  │
│  │   Sidebar   │    │   7 Input Tabs   │    │  Output   │  │
│  │             │    │                  │    │           │  │
│  │ • About     │    │ 🏠 Location      │    │ 💰 Price  │  │
│  │ • Metrics   │    │ 🏗️ Construction  │    │   Card    │  │
│  │ • Feature   │    │ 🧱 Basement      │    │           │  │
│  │   Importance│    │ 🛋️ Interior      │    │ 🔥 Feat.  │  │
│  │   Chart     │    │ 🚗 Garage        │    │   Impt.   │  │
│  │ • Quick     │    │ 🌳 Outdoor       │    │   Chart   │  │
│  │   Fill      │    │ 💰 Sale Info     │    │           │  │
│  └─────────────┘    └──────────────────┘    │ 📋 Table  │  │
│                                              │ ⬇️ CSV    │  │
│                                              └───────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
train.csv (1460 rows × 80 cols)
    │
    ▼
Data Cleaning           → Drop sparse cols, fill NAs (median/mode/domain)
    │
    ▼
Feature Engineering     → pd.get_dummies(drop_first=True) → 244 columns
    │
    ▼
Train / Test Split      → 80% / 20%, random_state=42
    │
    ▼
Model Training          → Linear Regression, Ridge, Lasso,
                          Decision Tree, Random Forest, Gradient Boosting
    │
    ▼
Cross-Validation        → 5-fold CV on GBR and top models
    │
    ▼
GridSearchCV Tuning     → n_estimators, learning_rate, max_depth, etc.
    │
    ▼
Best Model Saved        → gradient_boosting_model.pkl (joblib)
    │
    ▼
Streamlit App           → Reconstructs exact one-hot row → model.predict()
```

---

## 📊 Model Results

### Algorithm Comparison

| Model | Test R² | MAE ($) | Notes |
|---|---|---|---|
| **Gradient Boosting (tuned)** | **0.898** | **~$17,400** | ✅ Selected — best overall |
| Random Forest | ~0.875 | ~$20,000 | Strong but lower CV stability |
| Decision Tree | ~0.800 | ~$26,000 | Overfits without pruning |
| Ridge Regression | ~0.830 | ~$23,000 | Good baseline regularization |
| Lasso Regression | ~0.825 | ~$24,000 | Sparse features |
| Linear Regression | ~0.810 | ~$25,000 | Baseline |

### Best Hyperparameters (GridSearchCV)

```python
GradientBoostingRegressor(
    n_estimators    = 200,
    learning_rate   = 0.05,
    max_depth       = 3,
    min_samples_leaf = 1,
    min_samples_split = 2,
    random_state    = 42,
)
```

### Cross-Validation (5-Fold on Training Set)

```
Gradient Boosting   CV R² mean ≈ 0.868
Random Forest       CV R² mean ≈ 0.845
Decision Tree       CV R² mean ≈ 0.680
```

---

## 🗂️ Project Structure

```
House price predictor with GUI/
│
├── app.py                        # ✅ Streamlit web application (596+ lines)
├── house_price_analysis.ipynb    # ✅ Full ML notebook (EDA → Training → Tuning)
├── gradient_boosting_model.pkl   # ✅ Trained & serialized model
├── train.csv                     # 📊 Ames Housing Dataset (1460 rows)
├── requirements.txt              # 📦 Minimal dependencies (6 packages)
└── README.md                     # 📋 This file
```

---

## 🚀 Quick Start

### 1. Clone / Download

```bash
git clone https://github.com/fadifadifadifadifadi157-glitch/house-price-predictor.git
cd house-price-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Only 6 packages required — installs in under 60 seconds.

### 3. Run the App

```bash
streamlit run app.py
```

Your browser will open at **`http://localhost:8501`** automatically.

---

## 🧭 How to Use the App

1. **Sidebar — Quick Fill** *(optional)*: Choose a sample profile (Starter / Family / Luxury) to auto-populate sensible defaults for a quick demo.
2. **Tabs** — Fill in property details across the 7 tabs. Each tab groups related features:
   - `🏠 Location` — Zoning, lot size, neighborhood, proximity
   - `🏗️ Construction` — Year built, overall quality, roof, exterior
   - `🧱 Basement` — Finish type, square footage, bathrooms
   - `🛋️ Interior` — Living area, bedrooms, bathrooms, kitchen, utilities
   - `🚗 Garage` — Type, capacity, finish, area
   - `🌳 Outdoor` — Decks, porches, pool
   - `💰 Sale Info` — Month/year sold, sale type & condition
3. **Click "🔮 Predict Sale Price"** — get an instant price estimate powered by Gradient Boosting.
4. **View results** — see the predicted price card, key metrics, a feature importance chart, and a full feature table.
5. **Export** — download the full prediction (inputs + price) as a `.csv`.

---

## 🔧 Technical Details

### Encoding Strategy

The app **exactly mirrors** the notebook's `pd.get_dummies(drop_first=True)` + `.align()` encoding:

```python
# Notebook (training time):
X_train = pd.get_dummies(X_train, drop_first=True)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# App (inference time):
def build_input_row(numeric_values, categorical_values):
    row = pd.DataFrame(0, index=[0], columns=model.feature_names_in_)
    for feat, val in numeric_values.items():
        row.at[0, feat] = val
    for feat, val in categorical_values.items():
        dummy_col = f"{feat}_{val}"
        if dummy_col in row.columns:
            row.at[0, dummy_col] = 1
    return row  # 244-column DataFrame — matches model exactly
```

Choosing a "baseline" category (the first alphabetical value in `drop_first=True`) simply leaves all its dummy columns at 0 — identical to training behaviour.

### Key Dependencies

```
streamlit>=1.32   – Web UI framework
pandas>=2.0       – Data manipulation & one-hot encoding
numpy>=1.26       – Numerical operations
scikit-learn>=1.4 – GradientBoostingRegressor, metrics, model selection
joblib>=1.3       – Model serialization / deserialization
plotly>=5.20      – Interactive feature importance charts
```

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push this project to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** → select your repo → set **Main file path** to `app.py`.
4. Click **"Deploy"** — your app will be live in ~2 minutes.

> The `requirements.txt` is already Streamlit Cloud-compatible (minimal, no GPU packages).

---

## 📚 Dataset

**Ames Housing Dataset** — compiled by Dean De Cock for educational use.

- **Source**: [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Size**: 1,460 training samples × 79 original features
- **Target**: `SalePrice` — the sale price of the house in USD
- **Features used**: 74 (after dropping `Id`, `Alley`, `PoolQC`, `Fence`, `MiscFeature`)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">
Built with ❤️ using Python & Streamlit
</div>

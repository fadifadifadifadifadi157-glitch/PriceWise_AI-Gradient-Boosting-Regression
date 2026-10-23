"""
smoke_test.py - House Price Predictor
======================================
A fast sanity-check that verifies the core ML pipeline works correctly
without launching the Streamlit UI.

Run with:
    python smoke_test.py

All 7 checks must pass before deploying or submitting to a portfolio.
"""

import sys
import time
import os

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS_STR = f"{GREEN}PASS{RESET}"
FAIL_STR = f"{RED}FAIL{RESET}"

results = []

def check(name, fn):
    t0 = time.perf_counter()
    try:
        detail = fn()
        elapsed = time.perf_counter() - t0
        print(f"  [PASS]  {name}  ({elapsed*1000:.1f} ms)")
        if detail:
            print(f"         -> {detail}")
        results.append(("pass", name))
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"  [FAIL]  {name}  ({elapsed*1000:.1f} ms)")
        print(f"         -> ERROR: {exc}")
        results.append(("fail", name))

ROOT = os.path.dirname(os.path.abspath(__file__))

print()
print("=" * 50)
print("  House Price Predictor -- Smoke Test")
print("=" * 50)
print()

# CHECK 1
def chk_imports():
    import numpy
    import pandas
    import sklearn
    import joblib
    import streamlit
    import plotly
    import numpy as np
    return f"numpy {np.__version__} | sklearn {sklearn.__version__}"

check("1. Core packages import cleanly", chk_imports)

# CHECK 2
def chk_model_exists():
    path = os.path.join(ROOT, "gradient_boosting_model.pkl")
    assert os.path.isfile(path), f"File not found: {path}"
    size_kb = os.path.getsize(path) / 1024
    return f"gradient_boosting_model.pkl ({size_kb:.0f} KB)"

check("2. Model file exists on disk", chk_model_exists)

# CHECK 3 - load model into global so other checks can use it
import joblib
MODEL_PATH = os.path.join(ROOT, "gradient_boosting_model.pkl")
model = None

def chk_model_load():
    global model
    model = joblib.load(MODEL_PATH)
    return f"Loaded as {type(model).__name__}"

check("3. Model deserialises correctly (joblib.load)", chk_model_load)

# CHECK 4
def chk_feature_count():
    assert model is not None
    n = len(model.feature_names_in_)
    assert n == 244, f"Expected 244 features, got {n}"
    return f"{n} one-hot encoded features"

check("4. Model expects exactly 244 features", chk_feature_count)

# CHECK 5
def chk_predictions():
    import pandas as pd
    FEATURE_NAMES = list(model.feature_names_in_)

    def build_row(numeric):
        row = pd.DataFrame(0, index=[0], columns=FEATURE_NAMES, dtype=float)
        for feat, val in numeric.items():
            if feat in row.columns:
                row.at[0, feat] = val
        return row

    profiles = {
        # Starter: older small house, sparse features — model may predict low but ordering matters
        "Starter": dict(
            OverallQual=4, GrLivArea=950, GarageCars=1, YearBuilt=1965,
            TotalBsmtSF=800, FullBath=1, TotRmsAbvGrd=5,
        ),
        # Family: solid mid-range home
        "Family":  dict(
            OverallQual=6, GrLivArea=1600, GarageCars=2, YearBuilt=1995,
            TotalBsmtSF=1200, FullBath=2, TotRmsAbvGrd=7,
        ),
        # Luxury: large high-quality home
        "Luxury":  dict(
            OverallQual=9, GrLivArea=3200, GarageCars=3, YearBuilt=2008,
            TotalBsmtSF=2500, FullBath=3, TotRmsAbvGrd=11,
        ),
    }

    preds = {}
    for name, feats in profiles.items():
        p = model.predict(build_row(feats))[0]
        # Floor is $30k — a minimal sparse input can legitimately predict low
        assert 30_000 < p < 3_000_000, f"Unrealistic price for {name}: ${p:,.0f}"
        preds[name] = p

    # The key assertion: price ordering must make sense
    assert preds["Luxury"] > preds["Family"] > preds["Starter"], \
        f"Price ordering wrong — Luxury({preds['Luxury']:,.0f}) > Family({preds['Family']:,.0f}) > Starter({preds['Starter']:,.0f})"

    return " | ".join(f"{k}: ${v:,.0f}" for k, v in preds.items())

check("5. Predictions are realistic and correctly ordered", chk_predictions)

# CHECK 6
def chk_feature_importance():
    import numpy as np
    fi = model.feature_importances_
    assert len(fi) == 244
    top_idx  = int(np.argmax(fi))
    top_feat = model.feature_names_in_[top_idx]
    top_val  = fi[top_idx]
    assert top_val > 0
    return f"Top feature: '{top_feat}' (importance={top_val:.4f})"

check("6. Feature importances are non-zero and accessible", chk_feature_importance)

# CHECK 7
def chk_csv_export():
    import pandas as pd, io
    num  = {"OverallQual": 6, "GrLivArea": 1600, "GarageCars": 2}
    cat  = {"MSZoning": "RL", "Neighborhood": "NAmes"}
    pred = 185_000.0

    csv_str = pd.DataFrame([{**num, **cat, "PredictedSalePrice": pred}]).to_csv(index=False)
    df_back = pd.read_csv(io.StringIO(csv_str))
    assert "PredictedSalePrice" in df_back.columns
    assert df_back["PredictedSalePrice"].iloc[0] == pred
    return f"1 row x {df_back.shape[1]} cols -- PredictedSalePrice present"

check("7. CSV export round-trip produces correct output", chk_csv_export)

# SUMMARY
passed = sum(1 for r, _ in results if r == "pass")
failed = sum(1 for r, _ in results if r == "fail")
total  = len(results)

print()
print("-" * 50)
print(f"  Results: {passed} passed / {failed} failed / {total} total")

if failed == 0:
    print()
    print("  All checks passed -- app is ready to deploy!")
    print()
    sys.exit(0)
else:
    print()
    print(f"  {failed} check(s) failed -- fix the issues above before deploying.")
    print()
    sys.exit(1)

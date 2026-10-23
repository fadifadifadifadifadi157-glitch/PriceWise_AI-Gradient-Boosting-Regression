import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# MODERN CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background: radial-gradient(ellipse at 20% 0%, #1a1040 0%, #0a0d1a 55%, #050810 100%);
        min-height: 100vh;
    }

    /* ── HERO SECTION ── */
    .hero-wrap {
        position: relative;
        overflow: hidden;
        border-radius: 28px;
        margin-bottom: 2rem;
        padding: 2.8rem 3rem 2.4rem;
        background: linear-gradient(130deg, #1a0a3d 0%, #2d1260 40%, #1a0a3d 100%);
        border: 1px solid rgba(139, 92, 246, 0.35);
        box-shadow:
            0 0 0 1px rgba(139,92,246,0.15),
            0 32px 64px -20px rgba(109,40,217,0.5),
            inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .hero-wrap::before {
        content: '';
        position: absolute;
        top: -80px; right: -80px;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(167,139,250,0.25) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-wrap::after {
        content: '';
        position: absolute;
        bottom: -60px; left: 30%;
        width: 260px; height: 260px;
        background: radial-gradient(circle, rgba(236,72,153,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(139,92,246,0.2);
        border: 1px solid rgba(139,92,246,0.4);
        color: #c4b5fd;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 5px 14px;
        border-radius: 999px;
        margin-bottom: 1rem;
    }
    .hero-badge-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #a78bfa;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.7); }
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        color: #ffffff;
        font-size: 3rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0 0 0.5rem 0;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 50%, #f9a8d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        color: rgba(196,181,253,0.8);
        font-size: 1rem;
        font-weight: 400;
        line-height: 1.6;
        max-width: 600px;
        margin: 0;
    }
    .hero-stats {
        display: flex;
        gap: 1.5rem;
        margin-top: 1.8rem;
        flex-wrap: wrap;
    }
    .hero-stat {
        display: flex;
        flex-direction: column;
    }
    .hero-stat-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1;
    }
    .hero-stat-lbl {
        font-size: 0.72rem;
        color: rgba(196,181,253,0.7);
        font-weight: 500;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }
    .hero-divider {
        width: 1px;
        background: rgba(139,92,246,0.3);
        align-self: stretch;
    }

    /* ── GLASS CARDS ── */
    .glass-card {
        background: rgba(255,255,255,0.032);
        border: 1px solid rgba(255,255,255,0.07);
        border-left: 3px solid rgba(139,92,246,0.5);
        border-radius: 16px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(12px);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        border-left-color: rgba(167,139,250,0.85);
        box-shadow: 0 8px 30px -8px rgba(109,40,217,0.25);
    }

    /* ── SECTION LABELS ── */
    .section-label {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #a78bfa;
        font-weight: 700;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(139,92,246,0.15);
    }
    .section-label::before {
        content: '';
        display: inline-block;
        width: 4px; height: 16px;
        border-radius: 2px;
        background: linear-gradient(180deg, #a78bfa, #ec4899);
        flex-shrink: 0;
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255,255,255,0.025);
        padding: 6px 8px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        color: rgba(196,181,253,0.7);
        font-weight: 500;
        font-size: 0.88rem;
        padding: 9px 18px;
        transition: background 0.2s ease, color 0.2s ease;
        border: 1px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #c4b5fd;
        background: rgba(139,92,246,0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
        color: white !important;
        box-shadow: 0 4px 14px -4px rgba(124,58,237,0.7);
        border-color: rgba(167,139,250,0.3) !important;
    }

    /* ── PREDICT BUTTON ── */
    div.stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #ec4899 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        border: none;
        border-radius: 16px;
        padding: 0.85rem 1.5rem;
        width: 100%;
        letter-spacing: 0.3px;
        box-shadow:
            0 0 0 1px rgba(167,139,250,0.3),
            0 16px 40px -12px rgba(124,58,237,0.7);
        transition: transform 0.18s cubic-bezier(.34,1.56,.64,1), box-shadow 0.18s ease;
        cursor: pointer;
    }
    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow:
            0 0 0 1px rgba(167,139,250,0.5),
            0 20px 50px -12px rgba(124,58,237,0.85);
        color: white;
    }
    div.stButton > button:active {
        transform: translateY(0px) scale(0.99);
    }

    /* ── RESULT CARD ── */
    .result-card {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #064e3b 0%, #065f46 40%, #0369a1 100%);
        border: 1px solid rgba(52,211,153,0.3);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        text-align: center;
        box-shadow:
            0 0 0 1px rgba(52,211,153,0.15),
            0 24px 60px -16px rgba(16,185,129,0.5);
        margin-top: 1.5rem;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: -40px; left: 50%; transform: translateX(-50%);
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(52,211,153,0.2) 0%, transparent 70%);
        pointer-events: none;
    }
    .result-label {
        color: rgba(167,243,208,0.9);
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .result-price {
        font-family: 'Space Grotesk', sans-serif;
        color: #ffffff;
        font-size: 3.6rem;
        font-weight: 700;
        letter-spacing: -2px;
        margin: 0;
        line-height: 1;
        text-shadow: 0 0 40px rgba(52,211,153,0.4);
    }
    .result-sub {
        color: rgba(167,243,208,0.7);
        font-size: 0.82rem;
        margin-top: 0.8rem;
    }

    /* ── METRIC CHIPS (sidebar) ── */
    .chip-grid {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin: 0.5rem 0;
    }
    .chip {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(139,92,246,0.1);
        color: #e8e8f5;
        padding: 10px 16px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid rgba(139,92,246,0.25);
        transition: background 0.2s ease;
    }
    .chip:hover {
        background: rgba(139,92,246,0.18);
    }
    .chip-icon {
        font-size: 1rem;
        flex-shrink: 0;
    }
    .chip-val {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        color: #a78bfa;
    }
    .chip-label {
        font-size: 0.75rem;
        color: rgba(196,181,253,0.7);
        font-weight: 400;
    }

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0f1e 0%, #12152a 100%);
        border-right: 1px solid rgba(139,92,246,0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-family: 'Space Grotesk', sans-serif;
        color: #c4b5fd;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: rgba(196,181,253,0.85);
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* ── INPUTS ── */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(139,92,246,0.2) !important;
        border-radius: 10px !important;
        color: #e8e8f5 !important;
        transition: border-color 0.2s ease;
    }
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div:focus-within {
        border-color: rgba(139,92,246,0.6) !important;
        box-shadow: 0 0 0 3px rgba(139,92,246,0.12) !important;
    }

    /* ── GENERAL TEXT ── */
    label, .stMarkdown p, .stCaption { color: #c4b5fd !important; }
    p, span, li { color: #d1d5e8; }

    /* ── INFO BOX ── */
    .stAlert {
        background: rgba(139,92,246,0.08) !important;
        border: 1px solid rgba(139,92,246,0.25) !important;
        border-radius: 14px !important;
        color: #c4b5fd !important;
    }

    /* ── DIVIDER ── */
    hr { border-color: rgba(139,92,246,0.15) !important; }

    /* ── QUICK-FILL HINT BAR ── */
    .hint-bar {
        display: flex;
        align-items: center;
        gap: 14px;
        background: linear-gradient(90deg, rgba(109,40,217,0.18) 0%, rgba(139,92,246,0.08) 100%);
        border: 1px solid rgba(139,92,246,0.3);
        border-left: 3px solid #a78bfa;
        border-radius: 14px;
        padding: 0.85rem 1.2rem;
        margin-bottom: 1.2rem;
        position: relative;
        overflow: hidden;
    }
    .hint-bar::after {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, rgba(167,139,250,0.04) 0%, transparent 60%);
        pointer-events: none;
    }
    .hint-arrow {
        font-size: 1.4rem;
        animation: slide-left 1.2s ease-in-out infinite;
        flex-shrink: 0;
        filter: drop-shadow(0 0 6px rgba(167,139,250,0.8));
    }
    @keyframes slide-left {
        0%, 100% { transform: translateX(0px);   opacity: 1; }
        50%       { transform: translateX(-8px);  opacity: 0.6; }
    }
    .hint-text {
        flex: 1;
        color: rgba(196,181,253,0.9);
        font-size: 0.88rem;
        font-weight: 400;
        line-height: 1.45;
    }
    .hint-text strong {
        color: #c4b5fd;
        font-weight: 700;
    }
    .hint-badge {
        display: inline-block;
        background: rgba(139,92,246,0.25);
        border: 1px solid rgba(167,139,250,0.4);
        color: #c4b5fd;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.8px;
        padding: 3px 10px;
        border-radius: 999px;
        white-space: nowrap;
        flex-shrink: 0;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "gradient_boosting_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
FEATURE_NAMES = list(model.feature_names_in_)

# ============================================================
# CATEGORY DEFINITIONS
# (derived exactly from the one-hot columns the trained model expects,
#  i.e. the same pd.get_dummies(..., drop_first=True) encoding used
#  in the notebook. The "first" alphabetical category in each list
#  below is the baseline that produces an all-zero dummy row.)
# ============================================================
CATEGORIES = {
    "MSZoning":      ["C (all)", "FV", "RH", "RL", "RM"],
    "Street":        ["Grvl", "Pave"],
    "LotShape":      ["IR1", "IR2", "IR3", "Reg"],
    "LandContour":   ["Bnk", "HLS", "Low", "Lvl"],
    "Utilities":     ["AllPub", "NoSeWa"],
    "LotConfig":     ["Corner", "CulDSac", "FR2", "FR3", "Inside"],
    "LandSlope":     ["Gtl", "Mod", "Sev"],
    "Neighborhood":  ["Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr",
                       "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel",
                       "NAmes", "NPkVill", "NWAmes", "NoRidge", "NridgHt", "OldTown",
                       "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker"],
    "Condition1":    ["Artery", "Feedr", "Norm", "PosA", "PosN", "RRAe", "RRAn", "RRNe", "RRNn"],
    "Condition2":    ["Artery", "Feedr", "Norm", "PosA", "PosN", "RRAe", "RRAn", "RRNn"],
    "BldgType":      ["1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"],
    "HouseStyle":    ["1.5Fin", "1.5Unf", "1Story", "2.5Fin", "2.5Unf", "2Story", "SFoyer", "SLvl"],
    "RoofStyle":     ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"],
    "RoofMatl":      ["ClyTile", "CompShg", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"],
    "Exterior1st":   ["AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd",
                       "HdBoard", "ImStucc", "MetalSd", "Plywood", "Stone", "Stucco",
                       "VinylSd", "Wd Sdng", "WdShing"],
    "Exterior2nd":   ["AsbShng", "AsphShn", "Brk Cmn", "BrkFace", "CBlock", "CmentBd",
                       "HdBoard", "ImStucc", "MetalSd", "Other", "Plywood", "Stone",
                       "Stucco", "VinylSd", "Wd Sdng", "Wd Shng"],
    "MasVnrType":    ["BrkCmn", "BrkFace", "None", "Stone"],
    "ExterQual":     ["Ex", "Fa", "Gd", "TA"],
    "ExterCond":     ["Ex", "Fa", "Gd", "Po", "TA"],
    "Foundation":    ["BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"],
    "BsmtQual":      ["Ex", "Fa", "Gd", "None", "TA"],
    "BsmtCond":      ["Fa", "Gd", "None", "Po", "TA"],
    "BsmtExposure":  ["Av", "Gd", "Mn", "No", "None"],
    "BsmtFinType1":  ["ALQ", "BLQ", "GLQ", "LwQ", "None", "Rec", "Unf"],
    "BsmtFinType2":  ["ALQ", "BLQ", "GLQ", "LwQ", "None", "Rec", "Unf"],
    "Heating":       ["Floor", "GasA", "GasW", "Grav", "OthW", "Wall"],
    "HeatingQC":     ["Ex", "Fa", "Gd", "Po", "TA"],
    "CentralAir":    ["N", "Y"],
    "Electrical":    ["FuseA", "FuseF", "FuseP", "SBrkr"],
    "KitchenQual":   ["Ex", "Fa", "Gd", "TA"],
    "Functional":    ["Maj1", "Maj2", "Min1", "Min2", "Mod", "Sev", "Typ"],
    "FireplaceQu":   ["Ex", "Fa", "Gd", "None", "Po", "TA"],
    "GarageType":    ["2Types", "Attchd", "Basment", "BuiltIn", "CarPort", "Detchd", "None"],
    "GarageFinish":  ["Fin", "None", "RFn", "Unf"],
    "GarageQual":    ["Ex", "Fa", "Gd", "None", "Po", "TA"],
    "GarageCond":    ["Ex", "Fa", "Gd", "None", "Po", "TA"],
    "PavedDrive":    ["N", "P", "Y"],
    "SaleType":      ["COD", "CWD", "Con", "ConLD", "ConLI", "ConLw", "New", "Oth", "WD"],
    "SaleCondition": ["Abnorml", "AdjLand", "Alloca", "Family", "Normal", "Partial"],
}

MSSUBCLASS_MAP = {
    20: "1-Story, 1946+", 30: "1-Story, Pre-1945", 40: "1-Story w/ Finished Attic",
    45: "1.5-Story Unfinished", 50: "1.5-Story Finished", 60: "2-Story, 1946+",
    70: "2-Story, Pre-1945", 75: "2.5-Story", 80: "Split/Multi-Level",
    85: "Split Foyer", 90: "Duplex", 120: "1-Story PUD, 1946+",
    150: "1.5-Story PUD", 160: "2-Story PUD, 1946+", 180: "PUD Multilevel",
    190: "2-Family Conversion",
}

QUALITY_MAP = {1: "1 - Very Poor", 2: "2 - Poor", 3: "3 - Fair", 4: "4 - Below Avg",
               5: "5 - Average", 6: "6 - Above Avg", 7: "7 - Good", 8: "8 - Very Good",
               9: "9 - Excellent", 10: "10 - Very Excellent"}

# ============================================================
# HELPERS
# ============================================================
def build_input_row(numeric_values: dict, categorical_values: dict) -> pd.DataFrame:
    """Reconstruct the exact one-hot encoded row the model expects,
    mirroring pd.get_dummies(..., drop_first=True).align(...) from the notebook."""
    row = pd.DataFrame(0, index=[0], columns=FEATURE_NAMES, dtype=float)
    for feat, val in numeric_values.items():
        if feat in row.columns:
            row.at[0, feat] = val
    for feat, val in categorical_values.items():
        dummy_col = f"{feat}_{val}"
        if dummy_col in row.columns:
            row.at[0, dummy_col] = 1
        # if the column doesn't exist, it's the baseline (dropped) category -> stays all 0
    return row


def num_input(label, key, minv, maxv, default, step=1, help=None):
    return st.number_input(label, min_value=minv, max_value=maxv, value=default,
                            step=step, key=key, help=help)


def cat_input(label, key, feature):
    options = CATEGORIES[feature]
    return st.selectbox(label, options, key=key,
                         index=options.index(st.session_state.get(key, options[0]))
                         if st.session_state.get(key) in options else 0)


# ============================================================
# SIDEBAR
# ============================================================
# ============================================================
# FEATURE IMPORTANCE HELPER
# ============================================================
@st.cache_data
def get_feature_importance(_model, top_n=15):
    """Return top-N feature importances as a sorted DataFrame."""
    importances = _model.feature_importances_
    feat_names = _model.feature_names_in_
    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)
    return fi_df


def render_feature_importance_chart(fi_df):
    """Render a horizontal Plotly bar chart styled to match the dark theme."""
    fig = go.Figure(go.Bar(
        x=fi_df["Importance"],
        y=fi_df["Feature"],
        orientation="h",
        marker=dict(
            color=fi_df["Importance"],
            colorscale=[[0, "#6a5cff"], [0.5, "#a06bff"], [1, "#ff7ac6"]],
            showscale=False,
        ),
        text=[f"{v:.3f}" for v in fi_df["Importance"]],
        textposition="outside",
        textfont=dict(color="#e8e8f5", size=10),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=40, t=10, b=10),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#c9b8ff", size=10),
            gridcolor="rgba(255,255,255,0.05)",
        ),
        xaxis=dict(
            tickfont=dict(color="#c9b8ff", size=9),
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
        ),
        height=340,
    )
    return fig


with st.sidebar:
    st.markdown("""<div style='display:flex;align-items:center;gap:10px;margin-bottom:0.2rem'>
        <span style='font-size:1.5rem'>🏡</span>
        <span style='font-family:Space Grotesk,sans-serif;font-size:1.15rem;font-weight:700;color:#c4b5fd;letter-spacing:-0.5px'>PriceWise AI</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(196,181,253,0.65);font-size:0.8rem;margin-top:0;margin-bottom:1rem'>ML-powered house price estimator</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.markdown("""
        <div class="chip-grid">
            <div class="chip">
                <span class="chip-icon">🎯</span>
                <div>
                    <div class="chip-val">R² = 0.898</div>
                    <div class="chip-label">Test accuracy</div>
                </div>
            </div>
            <div class="chip">
                <span class="chip-icon">🔁</span>
                <div>
                    <div class="chip-val">R² = 0.868</div>
                    <div class="chip-label">5-fold cross-validation</div>
                </div>
            </div>
            <div class="chip">
                <span class="chip-icon">📉</span>
                <div>
                    <div class="chip-val">$17,400</div>
                    <div class="chip-label">Mean absolute error</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔥 Top 15 Feature Importances")
    fi_df = get_feature_importance(model, top_n=15)
    st.plotly_chart(render_feature_importance_chart(fi_df), use_container_width=True, config={"displayModeBar": False})
    st.markdown("---")
    st.markdown("### ⚡ Quick Fill Demo")
    quick = st.selectbox("Load a sample profile", ["-- none --", "Modest starter home", "Typical family home", "Luxury home"])
    st.markdown("---")
    st.caption("Built with Streamlit · Gradient Boosting · Ames Dataset")

# ============================================================
# QUICK-FILL SAMPLE PROFILES
# ============================================================
SAMPLES = {
    "Modest starter home": dict(
        OverallQual=4, OverallCond=5, YearBuilt=1965, YearRemodAdd=1965, GrLivArea=950,
        TotalBsmtSF=800, GarageCars=1, GarageArea=280, FullBath=1, BedroomAbvGr=2,
    ),
    "Typical family home": dict(
        OverallQual=6, OverallCond=6, YearBuilt=1995, YearRemodAdd=2000, GrLivArea=1600,
        TotalBsmtSF=1000, GarageCars=2, GarageArea=480, FullBath=2, BedroomAbvGr=3,
    ),
    "Luxury home": dict(
        OverallQual=9, OverallCond=8, YearBuilt=2008, YearRemodAdd=2009, GrLivArea=3200,
        TotalBsmtSF=1800, GarageCars=3, GarageArea=850, FullBath=3, BedroomAbvGr=4,
    ),
}
if quick != "-- none --":
    for k, v in SAMPLES[quick].items():
        st.session_state[f"num_{k}"] = v

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">
        <span class="hero-badge-dot"></span>
        Gradient Boosting · Ames Housing Dataset · 1,460 Samples
    </div>
    <h1 class="hero-title">🏡 PriceWise AI</h1>
    <p class="hero-sub">
        Enter your property details across the tabs below and get an instant,
        ML-powered sale price estimate — trained on 244 engineered features
        with hyperparameter-tuned Gradient Boosting.
    </p>
    <div class="hero-stats">
        <div class="hero-stat">
            <span class="hero-stat-val">0.898</span>
            <span class="hero-stat-lbl">Test R²</span>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
            <span class="hero-stat-val">$17.4K</span>
            <span class="hero-stat-lbl">Avg. Error (MAE)</span>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
            <span class="hero-stat-val">244</span>
            <span class="hero-stat-lbl">Model Features</span>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
            <span class="hero-stat-val">6</span>
            <span class="hero-stat-lbl">Algorithms Tested</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

numeric_values = {}
categorical_values = {}

# ── Quick-fill hint banner ──
st.markdown("""
<div class="hint-bar">
    <span class="hint-arrow">◀️</span>
    <div class="hint-text">
        <strong>⚡ New here?</strong> &nbsp;Use the <strong>Quick Fill Demo</strong> in the
        sidebar ← to instantly load a sample property profile
        (Starter / Family / Luxury) — then tweak any field you like.
    </div>
    <span class="hint-badge">💈 Sidebar ←</span>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "🏠 Location", "🏗️ Construction", "🧱 Basement",
    "🛋️ Interior", "🚗 Garage", "🌳 Outdoor", "💰 Sale Info"
])

# ---------------- TAB 1: LOCATION & BASIC INFO ----------------
with tabs[0]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Property Class & Zoning</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        ms_label = st.selectbox("Building class (MSSubClass)", list(MSSUBCLASS_MAP.values()),
                                 key="num_MSSubClass_label")
        numeric_values["MSSubClass"] = [k for k, v in MSSUBCLASS_MAP.items() if v == ms_label][0]
        categorical_values["MSZoning"] = cat_input("Zoning classification", "cat_MSZoning", "MSZoning")
    with c2:
        numeric_values["LotFrontage"] = num_input("Lot frontage (linear feet of street)", "num_LotFrontage", 0, 400, 70)
        numeric_values["LotArea"] = num_input("Lot area (sq ft)", "num_LotArea", 500, 250000, 9500)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Lot Characteristics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        categorical_values["Street"] = cat_input("Street type", "cat_Street", "Street")
        categorical_values["LotShape"] = cat_input("Lot shape", "cat_LotShape", "LotShape")
    with c2:
        categorical_values["LandContour"] = cat_input("Land contour", "cat_LandContour", "LandContour")
        categorical_values["Utilities"] = cat_input("Utilities available", "cat_Utilities", "Utilities")
    with c3:
        categorical_values["LotConfig"] = cat_input("Lot configuration", "cat_LotConfig", "LotConfig")
        categorical_values["LandSlope"] = cat_input("Land slope", "cat_LandSlope", "LandSlope")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Neighborhood & Surroundings</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        categorical_values["Neighborhood"] = cat_input("Neighborhood", "cat_Neighborhood", "Neighborhood")
    with c2:
        categorical_values["Condition1"] = cat_input("Proximity condition 1", "cat_Condition1", "Condition1")
        categorical_values["BldgType"] = cat_input("Building type", "cat_BldgType", "BldgType")
    with c3:
        categorical_values["Condition2"] = cat_input("Proximity condition 2", "cat_Condition2", "Condition2")
        categorical_values["HouseStyle"] = cat_input("House style", "cat_HouseStyle", "HouseStyle")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 2: CONSTRUCTION & QUALITY ----------------
with tabs[1]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Overall Ratings & Age</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        oq_label = st.selectbox("Overall quality", list(QUALITY_MAP.values()), index=5, key="num_OverallQual_label")
        numeric_values["OverallQual"] = int(oq_label.split(" - ")[0])
    with c2:
        oc_label = st.selectbox("Overall condition", list(QUALITY_MAP.values()), index=5, key="num_OverallCond_label")
        numeric_values["OverallCond"] = int(oc_label.split(" - ")[0])
    with c3:
        pass
    c1, c2 = st.columns(2)
    with c1:
        numeric_values["YearBuilt"] = num_input("Year built", "num_YearBuilt", 1870, 2026, 1995)
    with c2:
        numeric_values["YearRemodAdd"] = num_input("Year remodeled/added", "num_YearRemodAdd", 1870, 2026, 2000)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Roof & Exterior</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        categorical_values["RoofStyle"] = cat_input("Roof style", "cat_RoofStyle", "RoofStyle")
        categorical_values["Exterior1st"] = cat_input("Exterior covering 1", "cat_Exterior1st", "Exterior1st")
    with c2:
        categorical_values["RoofMatl"] = cat_input("Roof material", "cat_RoofMatl", "RoofMatl")
        categorical_values["Exterior2nd"] = cat_input("Exterior covering 2", "cat_Exterior2nd", "Exterior2nd")
    with c3:
        categorical_values["ExterQual"] = cat_input("Exterior quality", "cat_ExterQual", "ExterQual")
        categorical_values["ExterCond"] = cat_input("Exterior condition", "cat_ExterCond", "ExterCond")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Masonry Veneer & Foundation</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        categorical_values["MasVnrType"] = cat_input("Masonry veneer type", "cat_MasVnrType", "MasVnrType")
    with c2:
        numeric_values["MasVnrArea"] = num_input("Masonry veneer area (sq ft)", "num_MasVnrArea", 0, 2000, 0)
    with c3:
        categorical_values["Foundation"] = cat_input("Foundation type", "cat_Foundation", "Foundation")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 3: BASEMENT ----------------
with tabs[2]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Basement Quality & Layout</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        categorical_values["BsmtQual"] = cat_input("Basement height/quality", "cat_BsmtQual", "BsmtQual")
        categorical_values["BsmtFinType1"] = cat_input("Basement finish type 1", "cat_BsmtFinType1", "BsmtFinType1")
    with c2:
        categorical_values["BsmtCond"] = cat_input("Basement condition", "cat_BsmtCond", "BsmtCond")
        categorical_values["BsmtFinType2"] = cat_input("Basement finish type 2", "cat_BsmtFinType2", "BsmtFinType2")
    with c3:
        categorical_values["BsmtExposure"] = cat_input("Basement exposure", "cat_BsmtExposure", "BsmtExposure")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Basement Area & Baths</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        numeric_values["BsmtFinSF1"] = num_input("Finished area type 1 (sq ft)", "num_BsmtFinSF1", 0, 6000, 400)
        numeric_values["BsmtFullBath"] = num_input("Basement full baths", "num_BsmtFullBath", 0, 4, 0)
    with c2:
        numeric_values["BsmtFinSF2"] = num_input("Finished area type 2 (sq ft)", "num_BsmtFinSF2", 0, 2000, 0)
        numeric_values["BsmtHalfBath"] = num_input("Basement half baths", "num_BsmtHalfBath", 0, 2, 0)
    with c3:
        numeric_values["BsmtUnfSF"] = num_input("Unfinished area (sq ft)", "num_BsmtUnfSF", 0, 3000, 400)
    numeric_values["TotalBsmtSF"] = num_input(
        "Total basement area (sq ft)", "num_TotalBsmtSF", 0, 7000,
        int(numeric_values["BsmtFinSF1"] + numeric_values["BsmtFinSF2"] + numeric_values["BsmtUnfSF"]),
        help="Typically Fin1 + Fin2 + Unfinished area"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 4: INTERIOR ----------------
with tabs[3]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Living Area</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        numeric_values["1stFlrSF"] = num_input("1st floor area (sq ft)", "num_1stFlrSF", 200, 5000, 1000)
    with c2:
        numeric_values["2ndFlrSF"] = num_input("2nd floor area (sq ft)", "num_2ndFlrSF", 0, 3000, 600)
    with c3:
        numeric_values["LowQualFinSF"] = num_input("Low-quality finished area (sq ft)", "num_LowQualFinSF", 0, 1000, 0)
    numeric_values["GrLivArea"] = num_input(
        "Above-grade living area (sq ft)", "num_GrLivArea", 300, 6000,
        int(numeric_values["1stFlrSF"] + numeric_values["2ndFlrSF"] + numeric_values["LowQualFinSF"])
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Rooms & Bathrooms</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        numeric_values["FullBath"] = num_input("Full baths (above grade)", "num_FullBath", 0, 5, 2)
        numeric_values["HalfBath"] = num_input("Half baths (above grade)", "num_HalfBath", 0, 3, 0)
    with c2:
        numeric_values["BedroomAbvGr"] = num_input("Bedrooms (above grade)", "num_BedroomAbvGr", 0, 10, 3)
        numeric_values["KitchenAbvGr"] = num_input("Kitchens (above grade)", "num_KitchenAbvGr", 0, 3, 1)
    with c3:
        numeric_values["TotRmsAbvGrd"] = num_input("Total rooms (above grade)", "num_TotRmsAbvGrd", 2, 16, 6)
        numeric_values["Fireplaces"] = num_input("Fireplaces", "num_Fireplaces", 0, 5, 0)
    with c4:
        categorical_values["KitchenQual"] = cat_input("Kitchen quality", "cat_KitchenQual", "KitchenQual")
        categorical_values["FireplaceQu"] = cat_input("Fireplace quality", "cat_FireplaceQu", "FireplaceQu")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Utilities & Systems</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        categorical_values["Heating"] = cat_input("Heating type", "cat_Heating", "Heating")
    with c2:
        categorical_values["HeatingQC"] = cat_input("Heating quality", "cat_HeatingQC", "HeatingQC")
    with c3:
        categorical_values["CentralAir"] = cat_input("Central air conditioning", "cat_CentralAir", "CentralAir")
    with c4:
        categorical_values["Electrical"] = cat_input("Electrical system", "cat_Electrical", "Electrical")
    categorical_values["Functional"] = cat_input("Home functionality rating", "cat_Functional", "Functional")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 5: GARAGE ----------------
with tabs[4]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Garage Details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        categorical_values["GarageType"] = cat_input("Garage type", "cat_GarageType", "GarageType")
        numeric_values["GarageYrBlt"] = num_input("Garage year built", "num_GarageYrBlt", 1870, 2026, 1998)
    with c2:
        categorical_values["GarageFinish"] = cat_input("Garage interior finish", "cat_GarageFinish", "GarageFinish")
        numeric_values["GarageCars"] = num_input("Garage capacity (cars)", "num_GarageCars", 0, 5, 2)
    with c3:
        numeric_values["GarageArea"] = num_input("Garage area (sq ft)", "num_GarageArea", 0, 1500, 480)
        categorical_values["PavedDrive"] = cat_input("Paved driveway", "cat_PavedDrive", "PavedDrive")
    c1, c2 = st.columns(2)
    with c1:
        categorical_values["GarageQual"] = cat_input("Garage quality", "cat_GarageQual", "GarageQual")
    with c2:
        categorical_values["GarageCond"] = cat_input("Garage condition", "cat_GarageCond", "GarageCond")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 6: OUTDOOR ----------------
with tabs[5]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Porches, Decks & Pool</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        numeric_values["WoodDeckSF"] = num_input("Wood deck area (sq ft)", "num_WoodDeckSF", 0, 1200, 0)
        numeric_values["OpenPorchSF"] = num_input("Open porch area (sq ft)", "num_OpenPorchSF", 0, 800, 0)
    with c2:
        numeric_values["EnclosedPorch"] = num_input("Enclosed porch area (sq ft)", "num_EnclosedPorch", 0, 800, 0)
        numeric_values["3SsnPorch"] = num_input("3-season porch area (sq ft)", "num_3SsnPorch", 0, 800, 0)
    with c3:
        numeric_values["ScreenPorch"] = num_input("Screen porch area (sq ft)", "num_ScreenPorch", 0, 800, 0)
        numeric_values["PoolArea"] = num_input("Pool area (sq ft)", "num_PoolArea", 0, 800, 0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Miscellaneous</div>', unsafe_allow_html=True)
    numeric_values["MiscVal"] = num_input("Misc. feature value ($)", "num_MiscVal", 0, 20000, 0)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 7: SALE INFO ----------------
with tabs[6]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Sale Timing & Terms</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        numeric_values["MoSold"] = num_input("Month sold", "num_MoSold", 1, 12, 6)
    with c2:
        numeric_values["YrSold"] = num_input("Year sold", "num_YrSold", 2006, 2026, 2008)
    with c3:
        categorical_values["SaleType"] = cat_input("Sale type", "cat_SaleType", "SaleType")
    with c4:
        categorical_values["SaleCondition"] = cat_input("Sale condition", "cat_SaleCondition", "SaleCondition")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PREDICT
# ============================================================
st.markdown("""
<div style='height:1px;background:linear-gradient(90deg,transparent,rgba(139,92,246,0.4),transparent);margin:1.5rem 0 1.2rem;'></div>
""", unsafe_allow_html=True)
predict_clicked = st.button("🔮  Predict Sale Price", use_container_width=True)

if predict_clicked:
    with st.spinner("🤖 Running Gradient Boosting inference..."):
        input_row = build_input_row(numeric_values, categorical_values)
        prediction = model.predict(input_row)[0]

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">✦ &nbsp; Estimated Sale Price &nbsp; ✦</div>
        <div class="result-price">${prediction:,.0f}</div>
        <div class="result-sub">Gradient Boosting · R² 0.898 · MAE ≈ $17,400</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Overall Quality", f"{numeric_values['OverallQual']} / 10")
    with c2:
        st.metric("Living Area", f'{int(numeric_values["GrLivArea"]):,} sq ft')
    with c3:
        st.metric("Garage Capacity", f'{int(numeric_values["GarageCars"])} car(s)')

    with st.expander("🔥 Feature Importance — What drives the prediction?", expanded=True):
        st.markdown(
            '<div class="section-label" style="margin-bottom:0.8rem">'
            'Top 15 features by Gradient Boosting importance</div>',
            unsafe_allow_html=True,
        )
        fi_df_main = get_feature_importance(model, top_n=15)
        st.plotly_chart(
            render_feature_importance_chart(fi_df_main),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with st.expander("📋 View all submitted feature values"):
        display_df = pd.DataFrame({
            "Feature": list(numeric_values.keys()) + list(categorical_values.keys()),
            "Value": [str(v) for v in list(numeric_values.values()) + list(categorical_values.values())]
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = pd.DataFrame([{**numeric_values, **categorical_values, "PredictedSalePrice": prediction}]).to_csv(index=False)
    st.download_button("⬇️ Download this prediction as CSV", csv, file_name="house_price_prediction.csv", mime="text/csv")
else:
    st.markdown("""
    <div style='
        background: rgba(139,92,246,0.06);
        border: 1px solid rgba(139,92,246,0.2);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        color: rgba(196,181,253,0.85);
        font-size: 0.95rem;
        text-align: center;
    '>
        👆 &nbsp; Fill in the property details across the tabs above, then click <strong style='color:#c4b5fd'>Predict Sale Price</strong>.
    </div>
    """, unsafe_allow_html=True)

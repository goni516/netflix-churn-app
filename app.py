from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "netflix_customer_churn.csv"
OPERATING_THRESHOLD = 0.54

FEATURES = [
    "avg_watch_time_per_day",
    "watch_hours",
    "last_login_days",
    "number_of_profiles",
    "payment_method",
    "subscription_type",
    "age",
    "monthly_fee",
]
NUMERIC_FEATURES = [
    "avg_watch_time_per_day",
    "watch_hours",
    "last_login_days",
    "number_of_profiles",
    "age",
    "monthly_fee",
]
CATEGORICAL_FEATURES = ["payment_method", "subscription_type"]
TARGET = "churned"

DEFAULT_ROW = {
    "avg_watch_time_per_day": 0.35,
    "watch_hours": 4.5,
    "last_login_days": 25,
    "number_of_profiles": 1,
    "payment_method": "Gift Card",
    "subscription_type": "Basic",
    "age": 51,
    "monthly_fee": 8.99,
}


st.set_page_config(
    page_title="Netflix Churn Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1120px;
        padding-top: 2.1rem;
        padding-bottom: 3rem;
    }
    .small-muted {
        color: #667085;
        font-size: 0.95rem;
        margin-top: -0.35rem;
    }
    .status-box {
        border: 1px solid #D0D5DD;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        background: #FFFFFF;
    }
    .risk-high {
        border-left: 6px solid #B42318;
        background: #FFF6F5;
    }
    .risk-medium {
        border-left: 6px solid #B54708;
        background: #FFFAEB;
    }
    .risk-low {
        border-left: 6px solid #027A48;
        background: #F6FEF9;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset=["customer_id"]).copy()
    x = df[FEATURES]
    y = df[TARGET].astype(int)

    x_train, _x_test, y_train, _y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model, len(df), float(y.mean())


def risk_level(probability: float) -> tuple[str, str]:
    if probability >= 0.70:
        return "HIGH", "risk-high"
    if probability >= OPERATING_THRESHOLD:
        return "MEDIUM", "risk-medium"
    return "LOW", "risk-low"


def make_input_frame(values: dict) -> pd.DataFrame:
    row = DEFAULT_ROW.copy()
    row.update(values)
    return pd.DataFrame([row], columns=FEATURES)


def predict_frame(model, frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    for col, default in DEFAULT_ROW.items():
        if col not in clean.columns:
            clean[col] = default
    clean = clean[FEATURES]
    for col in NUMERIC_FEATURES:
        clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(DEFAULT_ROW[col])

    probabilities = model.predict_proba(clean)[:, 1]
    result = clean.copy()
    result["churn_probability"] = probabilities
    result["risk_level"] = [risk_level(p)[0] for p in probabilities]
    result["retention_target"] = probabilities >= OPERATING_THRESHOLD
    return result


def explain_reasons(row: pd.Series, probability: float) -> list[str]:
    reasons = []
    if float(row["avg_watch_time_per_day"]) < 0.75:
        reasons.append("Daily watch time is low, which signals weak engagement.")
    if float(row["watch_hours"]) < 8:
        reasons.append("Total watch hours are low compared with active subscribers.")
    if float(row["last_login_days"]) >= 21:
        reasons.append("The customer has not logged in recently.")
    if row["subscription_type"] == "Basic":
        reasons.append("Basic-plan customers showed higher churn risk in the dataset.")
    if row["payment_method"] in {"Gift Card", "Crypto"}:
        reasons.append("This payment method was associated with higher churn in the training data.")
    if int(row["number_of_profiles"]) <= 1:
        reasons.append("Only one profile may indicate lower household or shared usage.")
    if not reasons:
        if probability >= OPERATING_THRESHOLD:
            reasons.append("The combined profile is above the retention threshold.")
        else:
            reasons.append("The customer profile is below the retention threshold.")
    return reasons[:4]


def render_single_prediction(model):
    st.subheader("Single Customer Prediction")
    with st.form("single_prediction"):
        col1, col2 = st.columns(2)
        with col1:
            avg_watch = st.number_input("Average Watch Time per Day", 0.0, 24.0, 0.35, 0.05)
            watch_hours = st.number_input("Total Watch Hours", 0.0, 100.0, 4.5, 0.5)
            last_login = st.number_input("Days Since Last Login", 0, 60, 25, 1)
            profiles = st.number_input("Number of Profiles", 1, 5, 1, 1)
        with col2:
            payment = st.selectbox("Payment Method", ["Gift Card", "Crypto", "PayPal", "Debit Card", "Credit Card"])
            subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
            age = st.number_input("Age", 13, 90, 51, 1)
            monthly_fee = st.selectbox("Monthly Fee", [8.99, 13.99, 17.99])
        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    values = {
        "avg_watch_time_per_day": avg_watch,
        "watch_hours": watch_hours,
        "last_login_days": last_login,
        "number_of_profiles": profiles,
        "payment_method": payment,
        "subscription_type": subscription,
        "age": age,
        "monthly_fee": monthly_fee,
    }
    if not submitted:
        return

    result = predict_frame(model, make_input_frame(values))
    probability = float(result.loc[0, "churn_probability"])
    level, css_class = risk_level(probability)

    st.markdown("---")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Churn Probability", f"{probability:.1%}")
    metric2.metric("Risk Level", level)
    metric3.metric("Retention Target", "Yes" if probability >= OPERATING_THRESHOLD else "No")
    st.progress(min(max(probability, 0.0), 1.0))

    st.markdown(
        f"""
        <div class="status-box {css_class}">
            <strong>Interpretation</strong><br>
            This customer is classified as <strong>{level}</strong> risk.
            The operating retention threshold is <strong>{OPERATING_THRESHOLD:.0%}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Key Signals")
    for reason in explain_reasons(result.loc[0], probability):
        st.write(f"- {reason}")

    if probability >= 0.70:
        action = "Offer a targeted retention incentive, personalized recommendations, or win-back message."
    elif probability >= OPERATING_THRESHOLD:
        action = "Monitor the customer and send a lighter engagement action."
    else:
        action = "Continue regular engagement; no urgent retention action is needed."
    st.info(f"Recommended action: {action}")


def render_batch_prediction(model):
    st.subheader("Batch CSV Prediction")
    st.write("Upload a CSV file with the selected model features to score multiple customers.")
    sample = pd.DataFrame([DEFAULT_ROW], columns=FEATURES)
    st.download_button(
        "Download CSV Template",
        sample.to_csv(index=False).encode("utf-8"),
        "netflix_churn_template.csv",
        "text/csv",
        use_container_width=True,
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        return
    result = predict_frame(model, pd.read_csv(uploaded))
    display = result.copy()
    display["churn_probability"] = display["churn_probability"].map(lambda value: f"{value:.1%}")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Predictions",
        result.to_csv(index=False).encode("utf-8"),
        "netflix_churn_predictions.csv",
        "text/csv",
        use_container_width=True,
    )


def main():
    model, row_count, churn_rate = train_model()
    st.title("Netflix Customer Churn Prediction System")
    st.markdown(
        "<p class='small-muted'>Predict churn risk for OTT subscribers using a Random Forest model.</p>",
        unsafe_allow_html=True,
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Training Rows", f"{row_count:,}")
    kpi2.metric("Dataset Churn Rate", f"{churn_rate:.1%}")
    kpi3.metric("Model", "Random Forest")
    kpi4.metric("Threshold", f"{OPERATING_THRESHOLD:.0%}")

    tab_predict, tab_batch, tab_notes = st.tabs(["Predict", "Batch CSV", "Model Notes"])
    with tab_predict:
        render_single_prediction(model)
    with tab_batch:
        render_batch_prediction(model)
    with tab_notes:
        st.subheader("Model Notes")
        st.write("The model is trained from the Netflix churn CSV included in this repository.")
        st.write("Important signals include average watch time per day, total watch hours, and days since last login.")
        st.dataframe(pd.DataFrame({"Feature": FEATURES}), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()

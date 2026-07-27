"""
Telco Customer Churn Prediction — Streamlit App
=================================================
Single-page app with:
  - Home / overview
  - Single customer prediction
  - Bulk prediction (CSV/Excel upload) with rich visual analysis
  - Sample data download (CSV, Excel, SQL)

Run with:
    streamlit run app.py

Place your trained pipeline as 'churn_pipeline.pkl' in the same folder.
"""

import io
import sqlite3
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — polished, modern UI
# ============================================================
st.markdown("""
<style>
    .main { background-color: #0e1117; }

    /* Header banner */
    .app-header {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }
    .app-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .app-header p {
        color: #eaeaff;
        font-size: 1.05rem;
        margin: 0;
    }

    /* KPI cards */
    .kpi-card {
        background: #161a23;
        border: 1px solid #2a2f3d;
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #9aa0ac;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Prediction result card */
    .result-churn {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 22px rgba(255,65,108,0.35);
    }
    .result-stay {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 22px rgba(56,239,125,0.35);
    }
    .result-churn h2, .result-stay h2 { margin: 0; font-size: 1.7rem; }
    .result-churn p, .result-stay p { margin: 0.3rem 0 0 0; opacity: 0.9; }

    /* Section titles */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.6rem;
        border-left: 5px solid #2575fc;
        padding-left: 0.6rem;
    }

    div[data-testid="stMetricValue"] { font-size: 1.6rem; }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS — column definitions based on the training data
# ============================================================
CATEGORICAL_OPTIONS = {
    "Gender": ["Male", "Female"],
    "Senior Citizen": ["Yes", "No"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "Phone Service": ["Yes", "No"],
    "Multiple Lines": ["Yes", "No", "No phone service"],
    "Internet Service": ["DSL", "Fiber optic", "No"],
    "Contract": ["Month-to-month", "Two year", "One year"],
    "Paperless Billing": ["Yes", "No"],
    "Payment Method": ["Mailed check", "Electronic check",
                        "Bank transfer (automatic)", "Credit card (automatic)"],
}

# Columns the trained pipeline expects as INPUT features.
# NOTE: Churn / Churn Value / Churn Score are target / derived columns and
# are excluded from the feature set used for prediction.
FEATURE_COLUMNS = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Tenure Months",
    "Phone Service", "Multiple Lines", "Internet Service", "Contract",
    "Paperless Billing", "Payment Method", "Monthly Charges",
    "Total Charges", "CLTV",
]

ALL_COLUMNS = FEATURE_COLUMNS + ["Churn", "Churn Value", "Churn Score"]

MODEL_PATH = "Model/Churn_pipeline.pkl"


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model(path):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, None
    except FileNotFoundError:
        return None, f"Model file '{path}' not found. Place it in the app folder."
    except Exception as e:
        return None, f"Error loading model: {e}"


model, model_error = load_model(MODEL_PATH)


def predict_churn(df_features: pd.DataFrame):
    """Returns (prediction_labels, probabilities) using the loaded pipeline."""
    preds = model.predict(df_features)
    try:
        probs = model.predict_proba(df_features)[:, 1]
    except Exception:
        probs = np.full(len(df_features), np.nan)
    return preds, probs


# ============================================================
# SAMPLE DATA GENERATION
# ============================================================
@st.cache_data
def generate_sample_data(n=25):
    rng = np.random.default_rng(42)
    data = {
        "Gender": rng.choice(CATEGORICAL_OPTIONS["Gender"], n),
        "Senior Citizen": rng.choice(CATEGORICAL_OPTIONS["Senior Citizen"], n, p=[0.16, 0.84]),
        "Partner": rng.choice(CATEGORICAL_OPTIONS["Partner"], n),
        "Dependents": rng.choice(CATEGORICAL_OPTIONS["Dependents"], n, p=[0.3, 0.7]),
        "Tenure Months": rng.integers(0, 73, n),
        "Phone Service": rng.choice(CATEGORICAL_OPTIONS["Phone Service"], n, p=[0.9, 0.1]),
        "Multiple Lines": rng.choice(CATEGORICAL_OPTIONS["Multiple Lines"], n),
        "Internet Service": rng.choice(CATEGORICAL_OPTIONS["Internet Service"], n),
        "Contract": rng.choice(CATEGORICAL_OPTIONS["Contract"], n),
        "Paperless Billing": rng.choice(CATEGORICAL_OPTIONS["Paperless Billing"], n),
        "Payment Method": rng.choice(CATEGORICAL_OPTIONS["Payment Method"], n),
        "Monthly Charges": np.round(rng.uniform(18.25, 118.75, n), 2),
    }
    df = pd.DataFrame(data)
    df["Total Charges"] = np.round(df["Monthly Charges"] * df["Tenure Months"] +
                                    rng.uniform(0, 50, n), 2)
    df["Churn"] = rng.choice(["Yes", "No"], n, p=[0.27, 0.73])
    df["Churn Value"] = (df["Churn"] == "Yes").astype(int)
    df["Churn Score"] = rng.integers(1, 101, n)
    df["CLTV"] = rng.integers(2000, 6500, n)
    return df[ALL_COLUMNS]


def to_excel_bytes(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="TelcoChurnSample")
    return buffer.getvalue()


def to_sql_bytes(df, table_name="telco_churn_sample"):
    """Generate a .sql file containing CREATE TABLE + INSERT statements."""
    buffer = io.StringIO()
    col_defs = []
    for col in df.columns:
        safe_col = col.replace(" ", "_")
        if pd.api.types.is_integer_dtype(df[col]):
            col_type = "INTEGER"
        elif pd.api.types.is_float_dtype(df[col]):
            col_type = "REAL"
        else:
            col_type = "TEXT"
        col_defs.append(f'    "{safe_col}" {col_type}')

    buffer.write(f"DROP TABLE IF EXISTS {table_name};\n")
    buffer.write(f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n);\n\n")

    for _, row in df.iterrows():
        values = []
        for val in row:
            if isinstance(val, str):
                escaped = val.replace("'", "''")
                values.append(f"'{escaped}'")
            elif pd.isna(val):
                values.append("NULL")
            else:
                values.append(str(val))
        cols_joined = ", ".join(f'"{c.replace(" ", "_")}"' for c in df.columns)
        buffer.write(f"INSERT INTO {table_name} ({cols_joined}) VALUES ({', '.join(values)});\n")

    return buffer.getvalue().encode("utf-8")


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="app-header">
    <h1>📡 Telco Customer Churn — Prediction Studio</h1>
    <p>Predict churn risk for individual customers or run bulk analysis on your entire customer base.</p>
</div>
""", unsafe_allow_html=True)

if model_error:
    st.warning(f"⚠️ {model_error}  The app UI will still work, but predictions require the model file.")

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home", "🔮 Single Prediction", "📊 Bulk Prediction & Analysis", "📥 Sample Data"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "This app uses a trained ML pipeline (`churn_pipeline.pkl`) to predict "
        "whether a telecom customer is likely to churn, based on demographic, "
        "account and service usage attributes."
    )
    st.markdown("### 🗂️ Feature columns used")
    st.code("\n".join(FEATURE_COLUMNS), language="text")


# ============================================================
# PAGE: HOME
# ============================================================
if page == "🏠 Home":
    col1, col2, col3, col4 = st.columns(4)
    for col, label, value, emoji in zip(
        [col1, col2, col3, col4],
        ["Model Status", "Feature Count", "Contract Types", "Payment Methods"],
        ["Loaded ✅" if model else "Not Found ❌", len(FEATURE_COLUMNS),
         len(CATEGORICAL_OPTIONS["Contract"]), len(CATEGORICAL_OPTIONS["Payment Method"])],
        ["🤖", "🧬", "📄", "💳"],
    ):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{emoji} {value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>How to use this app</div>", unsafe_allow_html=True)
    st.markdown("""
1. **🔮 Single Prediction** — Fill in one customer's details in a form and get an instant churn risk prediction.
2. **📊 Bulk Prediction & Analysis** — Upload a CSV/Excel file of many customers to get predictions for all of them plus interactive visual analysis (churn distribution, contract impact, charges vs churn, CLTV segments, correlations, etc.).
3. **📥 Sample Data** — Download a ready-made sample dataset in **CSV**, **Excel**, or **SQL** format matching the exact columns your model expects — useful for testing the bulk upload feature.
    """)

    st.markdown("<div class='section-title'>Expected data columns</div>", unsafe_allow_html=True)
    sample_preview = generate_sample_data(8)
    st.dataframe(sample_preview, use_container_width=True)


# ============================================================
# PAGE: SINGLE PREDICTION
# ============================================================
elif page == "🔮 Single Prediction":
    st.markdown("<div class='section-title'>Enter customer details</div>", unsafe_allow_html=True)

    with st.form("single_prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["Gender"])
            senior = st.selectbox("Senior Citizen", CATEGORICAL_OPTIONS["Senior Citizen"])
            partner = st.selectbox("Partner", CATEGORICAL_OPTIONS["Partner"])
            dependents = st.selectbox("Dependents", CATEGORICAL_OPTIONS["Dependents"])
            tenure = st.slider("Tenure (Months)", 0, 72, 12)

        with c2:
            phone_service = st.selectbox("Phone Service", CATEGORICAL_OPTIONS["Phone Service"])
            multiple_lines = st.selectbox("Multiple Lines", CATEGORICAL_OPTIONS["Multiple Lines"])
            internet_service = st.selectbox("Internet Service", CATEGORICAL_OPTIONS["Internet Service"])
            contract = st.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"])
            paperless = st.selectbox("Paperless Billing", CATEGORICAL_OPTIONS["Paperless Billing"])

        with c3:
            payment_method = st.selectbox("Payment Method", CATEGORICAL_OPTIONS["Payment Method"])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=65.0, step=0.5)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=20000.0, value=780.0, step=1.0)
            cltv = st.number_input("CLTV", min_value=0, max_value=10000, value=4000, step=50)

        submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "Gender": gender,
            "Senior Citizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure Months": tenure,
            "Phone Service": phone_service,
            "Multiple Lines": multiple_lines,
            "Internet Service": internet_service,
            "Contract": contract,
            "Paperless Billing": paperless,
            "Payment Method": payment_method,
            "Monthly Charges": monthly_charges,
            "Total Charges": total_charges,
            "CLTV": cltv,
        }])[FEATURE_COLUMNS]

        st.markdown("<div class='section-title'>Input summary</div>", unsafe_allow_html=True)
        st.dataframe(input_df, use_container_width=True)

        if model is None:
            st.error("Model not loaded — cannot generate a prediction. Add `churn_pipeline.pkl` to the app folder.")
        else:
            try:
                preds, probs = predict_churn(input_df)
                pred_label = preds[0]
                prob = probs[0] if not np.isnan(probs[0]) else None

                is_churn = str(pred_label) in ("Yes", "1", "1.0")
                st.markdown("<div class='section-title'>Prediction result</div>", unsafe_allow_html=True)

                res_col, gauge_col = st.columns([1, 1])
                with res_col:
                    if is_churn:
                        st.markdown(f"""
                        <div class="result-churn">
                            <h2>⚠️ Likely to Churn</h2>
                            <p>{f'Churn probability: {prob*100:.1f}%' if prob is not None else ''}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-stay">
                            <h2>✅ Likely to Stay</h2>
                            <p>{f'Churn probability: {prob*100:.1f}%' if prob is not None else ''}</p>
                        </div>
                        """, unsafe_allow_html=True)

                with gauge_col:
                    if prob is not None:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob * 100,
                            title={'text': "Churn Risk (%)"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#ff416c" if is_churn else "#38ef7d"},
                                'steps': [
                                    {'range': [0, 40], 'color': "#1f2430"},
                                    {'range': [40, 70], 'color': "#2a2f3d"},
                                    {'range': [70, 100], 'color': "#3a2030"},
                                ],
                            }
                        ))
                        fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20),
                                           paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                        st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.markdown("""
                            <div class="result-stay">
                            <h2>❌ Unable to Predict</h2>
                            <p>Please check that the uploaded model and input data match.</p>
                            </div>
                            """, unsafe_allow_html=True)


# ============================================================
# PAGE: BULK PREDICTION & ANALYSIS
# ============================================================
elif page == "📊 Bulk Prediction & Analysis":
    st.markdown("<div class='section-title'>Upload customer data</div>", unsafe_allow_html=True)
    st.caption("Accepted formats: CSV, XLSX. Columns should match the sample file (see 📥 Sample Data tab).")

    uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                bulk_df = pd.read_csv(uploaded_file)
            else:
                bulk_df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            bulk_df = None

        if bulk_df is not None:
            st.success(f"Loaded {bulk_df.shape[0]} rows and {bulk_df.shape[1]} columns.")
            with st.expander("Preview uploaded data", expanded=False):
                st.dataframe(bulk_df.head(20), use_container_width=True)

            missing_cols = [c for c in FEATURE_COLUMNS if c not in bulk_df.columns]
            if missing_cols:
                st.error(f"Uploaded file is missing required columns: {missing_cols}")
            elif model is None:
                st.error("Model not loaded — cannot generate predictions. Add `churn_pipeline.pkl` to the app folder.")
            else:
                with st.spinner("Running predictions..."):
                    try:
                        feature_df = bulk_df[FEATURE_COLUMNS].copy()
                        preds, probs = predict_churn(feature_df)
                        result_df = bulk_df.copy()
                        result_df["Predicted Churn"] = ["Yes" if str(p) in ("Yes", "1", "1.0") else "No" for p in preds]
                        result_df["Churn Probability (%)"] = np.round(probs * 100, 2) if not np.isnan(probs).all() else np.nan
                    except Exception:
                        st.error("Unable to generate churn predictions. Please check your model.")
                        result_df = None

                if result_df is not None:
                    # ---------------- KPI ROW ----------------
                    total_customers = len(result_df)
                    churn_count = (result_df["Predicted Churn"] == "Yes").sum()
                    churn_rate = churn_count / total_customers * 100 if total_customers else 0
                    avg_monthly = result_df["Monthly Charges"].mean() if "Monthly Charges" in result_df else 0
                    avg_cltv = result_df["CLTV"].mean() if "CLTV" in result_df else 0

                    k1, k2, k3, k4 = st.columns(4)
                    k1.markdown(f"""<div class="kpi-card"><div class="kpi-value">{total_customers}</div>
                                <div class="kpi-label">Total Customers</div></div>""", unsafe_allow_html=True)
                    k2.markdown(f"""<div class="kpi-card"><div class="kpi-value">{churn_count}</div>
                                <div class="kpi-label">Predicted Churners</div></div>""", unsafe_allow_html=True)
                    k3.markdown(f"""<div class="kpi-card"><div class="kpi-value">{churn_rate:.1f}%</div>
                                <div class="kpi-label">Churn Rate</div></div>""", unsafe_allow_html=True)
                    k4.markdown(f"""<div class="kpi-card"><div class="kpi-value">${avg_monthly:.0f}</div>
                                <div class="kpi-label">Avg Monthly Charges</div></div>""", unsafe_allow_html=True)

                    st.markdown("<div class='section-title'>Prediction results</div>", unsafe_allow_html=True)
                    st.dataframe(result_df, use_container_width=True)

                    csv_out = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download predictions as CSV", csv_out,
                                        "churn_predictions.csv", "text/csv")

                    # ---------------- VISUAL ANALYSIS ----------------
                    st.markdown("<div class='section-title'>📈 Visual analysis</div>", unsafe_allow_html=True)

                    tab1, tab2, tab3, tab4, tab5 = st.tabs(
                        ["Overview", "Demographics", "Services & Contract",
                         "Charges & CLTV", "Correlations"]
                    )

                    # --- Tab 1: Overview ---
                    with tab1:
                        col1, col2 = st.columns(2)
                        with col1:
                            fig = px.pie(result_df, names="Predicted Churn", hole=0.45,
                                         color="Predicted Churn",
                                         color_discrete_map={"Yes": "#ff416c", "No": "#38ef7d"},
                                         title="Predicted Churn Distribution")
                            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                            st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            if "Churn Probability (%)" in result_df and result_df["Churn Probability (%)"].notna().any():
                                fig = px.histogram(result_df, x="Churn Probability (%)", nbins=20,
                                                    color_discrete_sequence=["#2575fc"],
                                                    title="Churn Probability Distribution")
                                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                                st.plotly_chart(fig, use_container_width=True)

                    # --- Tab 2: Demographics ---
                    with tab2:
                        col1, col2 = st.columns(2)
                        with col1:
                            if "Gender" in result_df:
                                fig = px.histogram(result_df, x="Gender", color="Predicted Churn",
                                                    barmode="group", title="Churn by Gender",
                                                    color_discrete_map={"Yes": "#ff416c", "No": "#38ef7d"})
                                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                                st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            if "Senior Citizen" in result_df:
                                fig = px.histogram(result_df, x="Senior Citizen", color="Predicted Churn",
                                                    barmode="group", title="Churn by Senior Citizen Status",
                                                    color_discrete_map={"Yes": "#ff416c", "No": "#38ef7d"})
                                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                                st.plotly_chart(fig, use_container_width=True)

                        col3, col4 = st.columns(2)
                        with col3:
                            if "Partner" in result_df:
                                fig = px.histogram(result_df, x="Partner", color="Predicted Churn",
                                                    barmode="group", title="Churn by Partner Status",
                                                    color_discrete_map={"Yes": "#ff416c", "No": "#38ef7d"})
                                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                                st.plotly_chart(fig, use_container_width=True)
                        with col4:
                            if "Dependents" in result_df:
                                fig = px.histogram(result_df, x="Dependents", color="Predicted Churn",
                                                    barmode="group", title="Churn by Dependents",
                                                    color_discrete_map={"Yes": "#ff416c", "No": "#38ef7d"})
                                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                                st.plotly_chart(fig, use_container_width=True)

                    # --- Tab 3: Services & Contract ---
                    with tab3:
                        col1, col2 = st.columns(2)
                        with col1:
                            if "Contract" in result_df:
                                fig = px.histogram(result_df, x="Contract", color="Predicted Churn",
                                                    barmode="group", title="Churn by Contract Type",
                                                    color_discrete_map={"Yes": "#ff416c", "No": "#38ef7d"})
                                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                                st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            if "Internet Service" in result_df:
                                fig = px.histogram(result_df, x="Internet Service", color="Predicted Churn",
                                                    barmode="group", title="Churn by Internet Service",
                                                    color_discrete_map={"Yes": "#ff416c", "No": "#38ef7d"})
                                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                                st.plotly_chart(fig, use_container_width=True)

                        col3, col4 = st.columns(2)
                        with col3:
                            if "Payment Method" in result_df:
                                fig = px.histogram(result_df, x="Payment Method", color="Predicted Churn",
                                                    barmode="group", title="Churn by Payment Method",
                                                    color_discrete_map={"Yes": "#ff416c", "No": "#38ef7d"})
                                fig.update_xaxes(tickangle=20)
                                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                                st.plotly_chart(fig, use_container_width=True)
                        with col4:
                            if "Tenure Months" in result_df:
                                fig = px.histogram(result_df, x="Tenure Months", color="Predicted Churn",
                                                    nbins=20, title="Churn by Tenure (Months)",
                                                    color_discrete_map={"Yes": "#ff416c", "No": "#38ef7d"})
                                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                                st.plotly_chart(fig, use_container_width=True)

                    # --- Tab 4: Charges & CLTV ---
                    with tab4:
                        col1, col2 = st.columns(2)
                        with col1:
                            if "Monthly Charges" in result_df:
                                fig = px.box(result_df, x="Predicted Churn", y="Monthly Charges",
                                             color="Predicted Churn", title="Monthly Charges vs Churn",
                                             color_discrete_map={"Yes": "#ff416c", "No": "#38ef7d"})
                                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                                st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            if "Total Charges" in result_df:
                                fig = px.box(result_df, x="Predicted Churn", y="Total Charges",
                                             color="Predicted Churn", title="Total Charges vs Churn",
                                             color_discrete_map={"Yes": "#ff416c", "No": "#38ef7d"})
                                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                                st.plotly_chart(fig, use_container_width=True)

                        col3, col4 = st.columns(2)
                        with col3:
                            if "CLTV" in result_df:
                                fig = px.histogram(result_df, x="CLTV", color="Predicted Churn", nbins=25,
                                                    title="CLTV Distribution by Churn",
                                                    color_discrete_map={"Yes": "#ff416c", "No": "#38ef7d"})
                                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                                st.plotly_chart(fig, use_container_width=True)
                        with col4:
                            if "Monthly Charges" in result_df and "Tenure Months" in result_df:
                                fig = px.scatter(result_df, x="Tenure Months", y="Monthly Charges",
                                                  color="Predicted Churn", title="Tenure vs Monthly Charges",
                                                  color_discrete_map={"Yes": "#ff416c", "No": "#38ef7d"},
                                                  opacity=0.7)
                                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                                st.plotly_chart(fig, use_container_width=True)

                    # --- Tab 5: Correlations ---
                    with tab5:
                        numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
                        if len(numeric_cols) >= 2:
                            corr = result_df[numeric_cols].corr()
                            fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                                             color_continuous_scale="RdBu_r",
                                             title="Correlation Heatmap (numeric features)")
                            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Not enough numeric columns to compute correlations.")
    else:
        st.info("👆 Upload a CSV or Excel file to run bulk predictions. "
                "Need a template? Check the **📥 Sample Data** tab.")


# ============================================================
# PAGE: SAMPLE DATA DOWNLOAD
# ============================================================
elif page == "📥 Sample Data":
    st.markdown("<div class='section-title'>Download sample dataset</div>", unsafe_allow_html=True)
    st.caption("Use this sample file to test the Bulk Prediction & Analysis feature. "
               "It contains the exact columns your model expects.")

    n_rows = st.slider("Number of sample rows", 10, 200, 25, step=5)
    sample_df = generate_sample_data(n_rows)

    st.dataframe(sample_df, use_container_width=True)

    csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
    excel_bytes = to_excel_bytes(sample_df)
    sql_bytes = to_sql_bytes(sample_df)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("⬇️ Download CSV", csv_bytes, "telco_churn_sample.csv",
                            "text/csv", use_container_width=True)
    with d2:
        st.download_button("⬇️ Download Excel", excel_bytes, "telco_churn_sample.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True)
    with d3:
        st.download_button("⬇️ Download SQL", sql_bytes, "telco_churn_sample.sql",
                            "text/plain", use_container_width=True)

    st.markdown("<div class='section-title'>Column reference</div>", unsafe_allow_html=True)
    ref_data = {
        "Column": ["Gender", "Senior Citizen", "Partner", "Dependents", "Tenure Months",
                   "Phone Service", "Multiple Lines", "Internet Service", "Contract",
                   "Paperless Billing", "Payment Method", "Monthly Charges", "Total Charges",
                   "Churn", "Churn Value", "Churn Score", "CLTV"],
        "Type": ["Categorical", "Categorical", "Categorical", "Categorical", "Numeric",
                 "Categorical", "Categorical", "Categorical", "Categorical",
                 "Categorical", "Categorical", "Numeric", "Numeric",
                 "Categorical (target)", "Numeric (0/1 target)", "Numeric", "Numeric"],
        "Values / Range": ["Male, Female", "Yes, No", "Yes, No", "Yes, No", "0–72",
                            "Yes, No", "Yes, No, No phone service", "DSL, Fiber optic, No",
                            "Month-to-month, One year, Two year", "Yes, No",
                            "Mailed check, Electronic check, Bank transfer (automatic), Credit card (automatic)",
                            "$18.25–$118.75", "Cumulative $ amount", "Yes, No", "0 or 1",
                            "1–100", "~2000–6500"],
    }
    st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)
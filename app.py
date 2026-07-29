import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Customer Churn Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "churn_model.pkl"
DATA_PATH = "data/cleaned_engineered_churn.csv"


# ----------------------------------------------------------------------------
# LOAD ARTEFAK (MODEL & DATA)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_model_bundle(path: str):
    bundle = joblib.load(path)
    return bundle


@st.cache_data(show_spinner="Memuat dataset...")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


try:
    bundle = load_model_bundle(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"File model '{MODEL_PATH}' tidak ditemukan. "
        "Jalankan notebook pelatihan terlebih dahulu untuk menghasilkan artefak ini."
    )
    st.stop()

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"File data '{DATA_PATH}' tidak ditemukan. "
        "Pastikan path menuju `cleaned_engineered_churn.csv` sudah benar."
    )
    st.stop()

FALLBACK_NUMERIC_FEATURES = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "products_number",
    "estimated_salary",
    "BalanceSalaryRatio",
    "TenureAgeRatio",
    "ProductsPerTenure",
]
FALLBACK_CATEGORICAL_FEATURES = [
    "country",
    "gender",
    "credit_card",
    "active_member",
    "ZeroBalanceFlag",
    "AgeGroup",
    "CreditScoreCategory",
    "EngagementSegment",
]

if isinstance(bundle, dict) and "pipeline" in bundle:
    pipeline = bundle["pipeline"]
    numeric_features = bundle.get("numeric_features", FALLBACK_NUMERIC_FEATURES)
    categorical_features = bundle.get("categorical_features", FALLBACK_CATEGORICAL_FEATURES)
    model_name = bundle.get("model_name", "Model")
else:
    pipeline = bundle
    numeric_features = FALLBACK_NUMERIC_FEATURES
    categorical_features = FALLBACK_CATEGORICAL_FEATURES
    model_name = type(pipeline.named_steps.get("clf", pipeline)).__name__
    st.sidebar.info(
        "⚠️ `churn_model.pkl` disimpan tanpa metadata (bukan dict bundle). "
        "Menggunakan daftar fitur default — pastikan urutan sesuai notebook."
    )

all_features = numeric_features + categorical_features


# ----------------------------------------------------------------------------
# HELPER: FEATURE ENGINEERING (harus konsisten dengan notebook)
# ----------------------------------------------------------------------------
def engineer_features(raw: dict) -> pd.DataFrame:
    """Replikasi persis logika feature engineering di notebook untuk 1 baris input baru."""
    d = dict(raw)

    d["ZeroBalanceFlag"] = int(d["balance"] == 0)
    d["BalanceSalaryRatio"] = (
        d["balance"] / d["estimated_salary"] if d["estimated_salary"] != 0 else 0.0
    )
    d["TenureAgeRatio"] = d["tenure"] / d["age"] if d["age"] != 0 else 0.0
    d["ProductsPerTenure"] = d["products_number"] / (d["tenure"] + 1)

    age = d["age"]
    if age <= 30:
        age_group = "18-30"
    elif age <= 45:
        age_group = "31-45"
    elif age <= 60:
        age_group = "46-60"
    else:
        age_group = "60+"
    d["AgeGroup"] = age_group

    cs = d["credit_score"]
    if cs <= 579:
        cs_cat = "Poor"
    elif cs <= 669:
        cs_cat = "Fair"
    elif cs <= 739:
        cs_cat = "Good"
    elif cs <= 799:
        cs_cat = "Very Good"
    else:
        cs_cat = "Excellent"
    d["CreditScoreCategory"] = cs_cat

    if d["active_member"] == 1 and d["products_number"] >= 2:
        seg = "Active-Multi Product"
    elif d["active_member"] == 1 and d["products_number"] == 1:
        seg = "Active-Single Product"
    elif d["active_member"] == 0 and d["products_number"] >= 2:
        seg = "Inactive-Multi Product"
    else:
        seg = "Inactive-Single Product"
    d["EngagementSegment"] = seg

    return pd.DataFrame([d])


@st.cache_resource(show_spinner="Menyiapkan SHAP explainer...")
def get_explainer(_pipeline, _df: pd.DataFrame):
    """Bangun SHAP explainer generik (mendukung tree-based & linear model)."""
    prep = _pipeline.named_steps["prep"]
    clf = _pipeline.named_steps["clf"]

    X_bg = _df.drop(columns=["churn"]).copy()
    for c in ["AgeGroup", "CreditScoreCategory"]:
        if c in X_bg.columns:
            X_bg[c] = X_bg[c].astype(str)

    X_bg = X_bg[all_features]
    X_bg_sample = X_bg.sample(min(100, len(X_bg)), random_state=42)

    X_bg_transformed = prep.transform(X_bg_sample)
    if hasattr(X_bg_transformed, "toarray"):
        X_bg_transformed = X_bg_transformed.toarray()

    feature_names = prep.get_feature_names_out()
    explainer = shap.Explainer(clf, X_bg_transformed, feature_names=feature_names)
    return explainer


def compute_shap_contribution(pipeline, explainer, X_input: pd.DataFrame) -> pd.DataFrame:
    """Hitung kontribusi SHAP untuk satu baris input dan kembalikan sbg DataFrame rapi."""
    prep = pipeline.named_steps["prep"]
    X_transformed = prep.transform(X_input[all_features])
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    sv = explainer(X_transformed)
    values = np.array(sv.values)

    if values.ndim == 3:
        values = values[:, :, -1]

    contrib = values[0]
    feature_names = prep.get_feature_names_out()

    result = pd.DataFrame({"feature": feature_names, "shap_value": contrib})
    result["abs_value"] = result["shap_value"].abs()
    result = result.sort_values("abs_value", ascending=False)
    return result


# ----------------------------------------------------------------------------
# SIDEBAR — FILTER INTERAKTIF
# ----------------------------------------------------------------------------
st.sidebar.title("🏦 Churn Dashboard")
st.sidebar.markdown(f"**Model aktif:** `{model_name}`")
st.sidebar.divider()
st.sidebar.header("🔍 Filter Data (Tab Overview)")

countries = sorted(df["country"].unique().tolist())
genders = sorted(df["gender"].unique().tolist())

f_country = st.sidebar.multiselect("Country", options=countries, default=countries)
f_gender = st.sidebar.multiselect("Gender", options=genders, default=genders)

age_min, age_max = int(df["age"].min()), int(df["age"].max())
f_age = st.sidebar.slider("Rentang Usia", age_min, age_max, (age_min, age_max))

f_active = st.sidebar.multiselect(
    "Status Keaktifan",
    options=[1, 0],
    format_func=lambda x: "Active Member" if x == 1 else "Non-Active Member",
    default=[1, 0],
)

filtered_df = df[
    df["country"].isin(f_country)
    & df["gender"].isin(f_gender)
    & df["age"].between(f_age[0], f_age[1])
    & df["active_member"].isin(f_active)
].copy()

st.sidebar.caption(f"{len(filtered_df):,} dari {len(df):,} nasabah ditampilkan.")


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("🏦 Bank Customer Churn — Dashboard & Prediksi")
st.caption(
    "Dashboard interaktif untuk memahami pendorong churn nasabah dan memprediksi "
    "risiko churn untuk nasabah baru menggunakan model Machine Learning."
)

tab1, tab2 = st.tabs(["📊 Overview & Metrik Bisnis", "🔮 Prediksi Nasabah & SHAP"])


# ============================================================================
# TAB 1 — OVERVIEW & METRIK BISNIS
# ============================================================================
with tab1:
    if filtered_df.empty:
        st.warning("Tidak ada data yang cocok dengan filter yang dipilih.")
    else:
        total_customers = len(filtered_df)
        churn_rate = filtered_df["churn"].mean()
        avg_balance = filtered_df["balance"].mean()
        avg_products = filtered_df["products_number"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Nasabah", f"{total_customers:,}")
        c2.metric("Churn Rate", f"{churn_rate:.1%}")
        c3.metric("Rata-rata Saldo", f"${avg_balance:,.0f}")
        c4.metric("Rata-rata Jumlah Produk", f"{avg_products:.2f}")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribusi Usia vs Churn")
            fig_age = px.histogram(
                filtered_df,
                x="age",
                color=filtered_df["churn"].map({0: "Non-Churn", 1: "Churn"}),
                nbins=30,
                barmode="overlay",
                color_discrete_map={"Non-Churn": "#2F5496", "Churn": "#C00000"},
                labels={"age": "Usia", "color": "Status"},
            )
            fig_age.update_layout(legend_title_text="Status")
            st.plotly_chart(fig_age, use_container_width=True)

        with col2:
            st.subheader("Churn Rate per Jumlah Produk")
            prod_churn = (
                filtered_df.groupby("products_number")["churn"]
                .mean()
                .reset_index()
                .rename(columns={"churn": "churn_rate"})
            )
            fig_prod = px.bar(
                prod_churn,
                x="products_number",
                y="churn_rate",
                text_auto=".1%",
                color="churn_rate",
                color_continuous_scale="Reds",
                labels={"products_number": "Jumlah Produk", "churn_rate": "Churn Rate"},
            )
            fig_prod.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False)
            st.plotly_chart(fig_prod, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Churn Rate: Active vs Non-Active Member")
            active_churn = (
                filtered_df.groupby("active_member")["churn"]
                .mean()
                .reset_index()
                .rename(columns={"churn": "churn_rate"})
            )
            active_churn["status"] = active_churn["active_member"].map(
                {1: "Active Member", 0: "Non-Active Member"}
            )
            fig_active = px.bar(
                active_churn,
                x="status",
                y="churn_rate",
                text_auto=".1%",
                color="status",
                color_discrete_map={
                    "Active Member": "#2F5496",
                    "Non-Active Member": "#C00000",
                },
                labels={"churn_rate": "Churn Rate", "status": "Status Keaktifan"},
            )
            fig_active.update_layout(yaxis_tickformat=".0%", showlegend=False)
            st.plotly_chart(fig_active, use_container_width=True)

        with col4:
            st.subheader("Churn Rate per Negara")
            country_churn = (
                filtered_df.groupby("country")["churn"]
                .mean()
                .reset_index()
                .rename(columns={"churn": "churn_rate"})
            )
            fig_country = px.bar(
                country_churn.sort_values("churn_rate", ascending=False),
                x="country",
                y="churn_rate",
                text_auto=".1%",
                color="churn_rate",
                color_continuous_scale="Blues",
                labels={"country": "Negara", "churn_rate": "Churn Rate"},
            )
            fig_country.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False)
            st.plotly_chart(fig_country, use_container_width=True)

        st.subheader("Saldo vs Estimasi Gaji (berwarna berdasarkan status Churn)")
        scatter_df = filtered_df.sample(min(2000, len(filtered_df)), random_state=42).copy()
        scatter_df["status"] = scatter_df["churn"].map({0: "Non-Churn", 1: "Churn"})

        fig_scatter = px.scatter(
            scatter_df,
            x="estimated_salary",
            y="balance",
            color="status",
            opacity=0.6,
            color_discrete_map={"Non-Churn": "#2F5496", "Churn": "#C00000"},
            labels={"estimated_salary": "Estimasi Gaji", "balance": "Saldo", "status": "Status"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ============================================================================
# TAB 2 — PREDIKSI NASABAH & ANALISIS SHAP
# ============================================================================
with tab2:
    st.subheader("Masukkan Data Nasabah Baru")

    with st.form("prediction_form"):
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            in_country = st.selectbox("Country", options=countries)
            in_gender = st.selectbox("Gender", options=genders)
            in_age = st.number_input("Age", min_value=18, max_value=100, value=40)
            in_credit_score = st.number_input(
                "Credit Score", min_value=300, max_value=900, value=650
            )

        with fc2:
            in_tenure = st.number_input("Tenure (tahun)", min_value=0, max_value=15, value=5)
            in_balance = st.number_input(
                "Balance", min_value=0.0, value=75000.0, step=1000.0, format="%.2f"
            )
            in_products = st.selectbox("Jumlah Produk", options=[1, 2, 3, 4])
            in_salary = st.number_input(
                "Estimated Salary", min_value=0.0, value=100000.0, step=1000.0, format="%.2f"
            )

        with fc3:
            in_credit_card = st.selectbox(
                "Punya Kartu Kredit?", options=[1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak"
            )
            in_active = st.selectbox(
                "Active Member?", options=[1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak"
            )

        submitted = st.form_submit_button("🔮 Prediksi Churn", use_container_width=True)

    if submitted:
        raw_input = {
            "credit_score": in_credit_score,
            "country": in_country,
            "gender": in_gender,
            "age": in_age,
            "tenure": in_tenure,
            "balance": in_balance,
            "products_number": in_products,
            "credit_card": in_credit_card,
            "active_member": in_active,
            "estimated_salary": in_salary,
        }

        X_input = engineer_features(raw_input)
        X_input = X_input[all_features]

        proba_churn = pipeline.predict_proba(X_input)[:, 1][0]
        pred_label = "CHURN" if proba_churn >= 0.5 else "NON-CHURN"

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric("Probabilitas Churn", f"{proba_churn:.1%}")
            if pred_label == "CHURN":
                st.error(f"⚠️ Prediksi: **{pred_label}** — nasabah berisiko tinggi meninggalkan bank.")
            else:
                st.success(f"✅ Prediksi: **{pred_label}** — nasabah cenderung bertahan.")

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba_churn * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#C00000" if proba_churn >= 0.5 else "#2F5496"},
                        "steps": [
                            {"range": [0, 50], "color": "#E8F0FE"},
                            {"range": [50, 100], "color": "#FBE9E7"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.8,
                            "value": 50,
                        },
                    },
                    title={"text": "Risk Score"},
                )
            )
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("#### Kontribusi Fitur terhadap Prediksi (SHAP)")
            try:
                explainer = get_explainer(pipeline, df)
                shap_df = compute_shap_contribution(pipeline, explainer, X_input)
                top_n = shap_df.head(10).sort_values("shap_value")

                fig_shap = px.bar(
                    top_n,
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    color="shap_value",
                    color_continuous_scale=["#2F5496", "#F2F2F2", "#C00000"],
                    color_continuous_midpoint=0,
                    labels={"shap_value": "Kontribusi SHAP", "feature": "Fitur"},
                )
                fig_shap.update_layout(coloraxis_showscale=False, height=420)
                st.plotly_chart(fig_shap, use_container_width=True)
                st.caption(
                    "🔴 Merah mendorong prediksi ke arah **CHURN**, "
                    "🔵 biru mendorong prediksi ke arah **NON-CHURN**."
                )
            except Exception as e:
                st.info(
                    "Analisis SHAP tidak tersedia untuk model ini "
                    f"({model_name}). Detail: {e}"
                )

        with st.expander("Lihat data fitur hasil rekayasa (engineered features)"):
            st.dataframe(X_input.T.rename(columns={0: "value"}), use_container_width=True)
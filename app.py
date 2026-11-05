import json
from html import escape
from pathlib import Path

import requests
import streamlit as st


st.set_page_config(
    page_title="IDH Prediction System",
    page_icon="🏥",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"
MODEL_INFO_PATH = Path(__file__).with_name("idh_model_info.json")
SEX_CODES = {"Male": 0, "Female": 1}

st.markdown(
    """
    <style>
    :root {
        --ink: #153247;
        --muted: #647b8a;
        --teal: #087e83;
        --teal-dark: #075e64;
        --line: #e2ebef;
        --surface: #ffffff;
    }

    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Aptos', sans-serif;
        color: var(--ink) !important;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 92% 0%, rgba(196, 235, 230, 0.32), transparent 28%),
            #f5f8fa;
    }

    [data-testid="stHeader"] {
        background: rgba(245, 248, 250, 0.8);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1240px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        font-family: 'Segoe UI', 'Aptos', sans-serif;
        letter-spacing: -0.035em;
    }

    h1 {
        font-size: clamp(2rem, 3vw, 2.8rem) !important;
        font-weight: 800 !important;
    }

    h2, h3 {
        font-weight: 700 !important;
    }

    p, [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    [data-testid="stSidebar"] {
        background: #102c3e;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #e7f2f4;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #b7cbd3;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.45rem;
    }

    [data-testid="stSidebar"] [role="radio"] {
        border-radius: 10px;
        padding: 0.35rem 0.55rem;
    }

    [data-testid="stSidebar"] [role="radio"]:hover {
        background: rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] [role="radio"][aria-checked="true"] {
        background: rgba(80, 190, 177, 0.18);
        outline: 1px solid rgba(119, 218, 204, 0.25);
    }

    [data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.5rem 1.5rem 1rem;
        box-shadow: 0 12px 35px rgba(26, 60, 78, 0.05);
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 8px 22px rgba(26, 60, 78, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Segoe UI', 'Aptos', sans-serif;
    }

    .stButton > button, [data-testid="stFormSubmitButton"] > button {
        background: var(--teal);
        border: 1px solid var(--teal);
        border-radius: 10px;
        color: white;
        font-weight: 700;
        min-height: 2.8rem;
        transition: background 150ms ease, transform 150ms ease;
    }

    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
        background: var(--teal-dark);
        border-color: var(--teal-dark);
        color: white;
        transform: translateY(-1px);
    }

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] [role="combobox"] {
        border-radius: 9px;
        border-color: #d7e3e8;
        background: #fbfdfe;
    }

    .brand-block {
        padding: 0.5rem 0.15rem 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        margin-bottom: 1.25rem;
    }

    .brand-icon {
        display: inline-flex;
        width: 42px;
        height: 42px;
        align-items: center;
        justify-content: center;
        border-radius: 13px;
        background: linear-gradient(145deg, #1ba6a2, #0d7379);
        font-size: 1.35rem;
        margin-bottom: 0.8rem;
    }

    .brand-title {
        color: #f3fbfc;
        font: 800 1rem 'Segoe UI', 'Aptos', sans-serif;
        letter-spacing: -0.02em;
    }

    .brand-subtitle {
        color: #a9c2ca;
        font-size: 0.77rem;
        margin-top: 0.25rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 24px;
        padding: clamp(2rem, 5vw, 4.2rem);
        color: #f3fbfc;
        background: linear-gradient(118deg, #102f43 0%, #145661 58%, #087e83 100%);
        box-shadow: 0 24px 55px rgba(16, 55, 74, 0.18);
        margin-bottom: 1.5rem;
    }

    .hero:after {
        content: "";
        position: absolute;
        right: -80px;
        top: -150px;
        width: 380px;
        height: 380px;
        border: 1px solid rgba(221, 255, 249, 0.18);
        border-radius: 50%;
        box-shadow: 0 0 0 42px rgba(221, 255, 249, 0.035),
                    0 0 0 84px rgba(221, 255, 249, 0.025);
        pointer-events: none;
    }

    .hero-kicker {
        color: #9de3d7;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    }

    .hero-title {
        position: relative;
        z-index: 1;
        max-width: 670px;
        color: #ffffff;
        font: 800 clamp(2rem, 4.7vw, 3.65rem)/1.08 'Segoe UI', 'Aptos', sans-serif;
        letter-spacing: -0.055em;
    }

    .hero-copy {
        position: relative;
        z-index: 1;
        max-width: 600px;
        color: #d1e6e9;
        font-size: 1.02rem;
        line-height: 1.7;
        margin-top: 1rem;
    }

    .section-eyebrow {
        color: var(--teal);
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .feature-card {
        height: 100%;
        min-height: 240px;
        padding: 1.35rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 9px 24px rgba(26, 60, 78, 0.04);
    }

    .feature-icon {
        font-size: 1.35rem;
        margin-bottom: 0.8rem;
    }

    .feature-title {
        color: var(--ink);
        font: 700 1rem 'Segoe UI', 'Aptos', sans-serif;
        margin-bottom: 0.45rem;
    }

    .feature-copy {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .status-pill {
        display: inline-block;
        padding: 0.38rem 0.72rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .status-online {
        color: #13694b;
        background: #e1f4eb;
    }

    .status-offline {
        color: #8a5a16;
        background: #fff1d7;
    }

    .feature-chip {
        display: inline-block;
        margin: 0.25rem 0.25rem 0.25rem 0;
        padding: 0.48rem 0.72rem;
        color: #315466;
        background: #edf4f6;
        border: 1px solid #e1ebee;
        border-radius: 8px;
        font-size: 0.84rem;
    }

    .disclaimer {
        padding: 1rem 1.1rem;
        border-left: 4px solid #d5a14a;
        border-radius: 0 10px 10px 0;
        background: #fff8e9;
        color: #705626;
        line-height: 1.6;
        margin-top: 1.4rem;
    }

    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.2rem 1rem 2rem;
        }
        .hero {
            border-radius: 18px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


def check_backend():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def render_home():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Dialysis care · Risk assessment</div>
            <div class="hero-title">A clearer view of IDH risk.</div>
            <div class="hero-copy">
                Explore an academic prediction prototype designed to assess
                intradialytic hypotension risk from patient and dialysis-session
                parameters.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    connected = check_backend()
    if connected:
        status_class = "status-online"
        status_text = "Prediction service online"
    else:
        status_class = "status-offline"
        status_text = "Prediction service offline"

    st.markdown(
        f'<span class="status-pill {status_class}">{status_text}</span>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-eyebrow">Explore the workspace</div>'
        '<h2 style="margin-top:0;color:#153247;font-size:1.75rem">'
        "Built for a focused workflow</h2>",
        unsafe_allow_html=True
    )
    home_col, predict_col, info_col = st.columns(3, gap="medium")
    with home_col:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🩺</div>
                <div class="feature-title">Patient assessment</div>
                <div class="feature-copy">
                    Enter patient and dialysis-session values in a clear,
                    structured form.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with predict_col:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Transparent model details</div>
                <div class="feature-copy">
                    Review model metadata, required features, and threshold
                    information.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with info_col:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🛡️</div>
                <div class="feature-title">Designed with context</div>
                <div class="feature-copy">
                    Important limitations and safe-use guidance stay visible
                    throughout the experience.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="disclaimer">
            <strong>Academic prototype only.</strong> This application is not
            clinically validated and must not be used as a diagnosis or to make
            treatment decisions.
        </div>
        """,
        unsafe_allow_html=True
    )


def render_prediction():
    st.markdown('<div class="section-eyebrow">Risk assessment</div>', unsafe_allow_html=True)
    st.title("Patient assessment")
    st.write("Provide the details for this dialysis session to request a prediction.")

    if check_backend():
        st.markdown(
            '<span class="status-pill status-online">Prediction service online</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-pill status-offline">Prediction service offline</span>',
            unsafe_allow_html=True
        )

    with st.form("prediction_form"):
        patient_col, treatment_col = st.columns(2)

        with patient_col:
            st.markdown("### Patient profile")
            sex_label = st.selectbox("Sex", list(SEX_CODES))
            D_Sexo = SEX_CODES[sex_label]
            st.caption(
                "Dropdown mapping: Male = 0, Female = 1. Confirm this matches "
                "the encoding used to train the model."
            )
            D_Edad = st.number_input("Age (D_Edad)", value=0.0)
            D_Dializador = st.number_input("Dialyzer (D_Dializador)", value=0.0)
            D_Bano = st.number_input("Dialysis bath (D_Bano)", value=0.0)
            D_Tecnica = st.number_input("Technique (D_Tecnica)", value=0.0)

            st.markdown("### Weight information")
            D_PesoSeco = st.number_input("Dry weight (D_PesoSeco)", value=0.0)
            D_PesoPre = st.number_input(
                "Pre-dialysis weight (D_PesoPre)",
                value=0.0
            )
            D_Ganancia = st.number_input(
                "Weight gain (D_Ganancia)",
                min_value=0.0,
                value=2.0
            )

        with treatment_col:
            st.markdown("### Dialysis settings")
            D_TemperaturaBano = st.number_input(
                "Dialysate temperature (D_TemperaturaBano)",
                min_value=0.0,
                value=36.0
            )
            D_FlujoSangre = st.number_input(
                "Blood flow (D_FlujoSangre)",
                min_value=0.0,
                value=300.0
            )
            D_FlujoBano = st.number_input(
                "Dialysate flow (D_FlujoBano)",
                min_value=0.0,
                value=500.0
            )
            D_ConductividadBano = st.number_input(
                "Dialysate conductivity (D_ConductividadBano)",
                min_value=0.0,
                value=14.0
            )
            D_ConductividadBicarbonato = st.number_input(
                "Bicarbonate conductivity (D_ConductividadBicarbonato)",
                min_value=0.0,
                value=3.0
            )

        submit = st.form_submit_button(
            "Predict IDH",
            use_container_width=True
        )

    if not submit:
        return

    patient_data = {
        "D_Sexo": int(D_Sexo),
        "D_Edad": float(D_Edad),
        "D_Dializador": int(D_Dializador),
        "D_Bano": int(D_Bano),
        "D_Tecnica": int(D_Tecnica),
        "D_PesoSeco": float(D_PesoSeco),
        "D_PesoPre": float(D_PesoPre),
        "D_Ganancia": float(D_Ganancia),
        "D_TemperaturaBano": float(D_TemperaturaBano),
        "D_FlujoSangre": float(D_FlujoSangre),
        "D_FlujoBano": float(D_FlujoBano),
        "D_ConductividadBano": float(D_ConductividadBano),
        "D_ConductividadBicarbonato": float(D_ConductividadBicarbonato)
    }

    try:
        with st.spinner("Predicting IDH..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=patient_data,
                timeout=30
            )

        if response.status_code != 200:
            st.error(f"Prediction failed: {response.text}")
            return

        result = response.json()
        probability = result["idh_probability"]
        prediction = result["prediction"]
        threshold = result.get("threshold", 0.35)

        st.divider()
        st.subheader("Prediction result")
        result_col, threshold_col = st.columns(2)
        result_col.metric("IDH probability", f"{probability * 100:.2f}%")
        threshold_col.metric("Decision threshold", f"{threshold * 100:.0f}%")

        if prediction == 1:
            st.error(
                "IDH predicted — the model identifies a risk of "
                "intradialytic hypotension."
            )
        else:
            st.success("IDH not predicted for these inputs.")

        st.caption(
            "This is an academic prediction prototype, not a clinically "
            "validated diagnostic tool."
        )
    except requests.exceptions.RequestException as error:
        st.error(f"Backend connection error: {error}")


def render_model_information():
    st.markdown('<div class="section-eyebrow">Behind the prediction</div>', unsafe_allow_html=True)
    st.title("Model information")
    st.write(
        "The following details are read from this project's model metadata."
    )

    with MODEL_INFO_PATH.open(encoding="utf-8") as model_info_file:
        model_info = json.load(model_info_file)

    model_col, threshold_col, dataset_col = st.columns(3)
    model_col.metric("Model", model_info["model_name"])
    threshold_col.metric("Metadata threshold", f"{model_info['threshold']:.0%}")
    dataset_col.metric("Training dataset", model_info["dataset"])

    st.subheader("Required inputs")
    feature_markup = "".join(
        f'<span class="feature-chip">{escape(feature)}</span>'
        for feature in model_info["features"]
    )
    st.markdown(feature_markup, unsafe_allow_html=True)

    st.subheader("Categorical inputs")
    categorical_markup = "".join(
        f'<span class="feature-chip">{escape(feature)}</span>'
        for feature in model_info["categorical_features"]
    )
    st.markdown(categorical_markup, unsafe_allow_html=True)

    st.caption(
        "The deployed API may apply its own threshold; the prediction result "
        "shows the threshold returned by the backend."
    )


def render_about():
    st.markdown('<div class="section-eyebrow">Purpose and safe use</div>', unsafe_allow_html=True)
    st.title("About this prototype")
    st.write(
        "This application demonstrates a machine-learning workflow for "
        "estimating intradialytic hypotension (IDH) risk from patient and "
        "dialysis-session inputs."
    )

    st.subheader("How to use it")
    st.markdown(
        "1. Open **Predict IDH** from the sidebar.\n"
        "2. Enter the requested patient and dialysis values.\n"
        "3. Submit the form to send those values to the local prediction API.\n"
        "4. Review the probability and threshold returned by the API."
    )

    st.subheader("Important limitation")
    st.markdown(
        """
        <div class="disclaimer">
            <strong>Important limitation.</strong> This is an academic
            prototype, not a clinically validated diagnostic tool. Predictions
            are informational only and must not replace professional clinical
            judgment.
        </div>
        """,
        unsafe_allow_html=True
    )


st.sidebar.markdown(
    """
    <div class="brand-block">
        <div class="brand-icon">✚</div>
        <div class="brand-title">IDH Care</div>
        <div class="brand-subtitle">Prediction workspace</div>
    </div>
    """,
    unsafe_allow_html=True
)
page_names = {
    "⌂   Overview": "Home",
    "✚   Patient assessment": "Predict IDH",
    "▤   Model information": "Model Information",
    "ⓘ   About": "About"
}
page_labels = list(page_names)
selected_page = st.sidebar.radio(
    "WORKSPACE",
    page_labels,
    key="current_page"
)

if page_names[selected_page] == "Home":
    render_home()
elif page_names[selected_page] == "Predict IDH":
    render_prediction()
elif page_names[selected_page] == "Model Information":
    render_model_information()
else:
    render_about()

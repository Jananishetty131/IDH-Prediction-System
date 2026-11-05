import json
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


def check_backend():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def render_home():
    st.title("🏥 IDH Prediction System")
    st.subheader("A decision-support prototype for dialysis care")
    st.write(
        "Explore patient-specific intradialytic hypotension (IDH) "
        "risk predictions, model details, and guidance for using this prototype."
    )

    connected = check_backend()
    if connected:
        st.success("Prediction service is online.")
    else:
        st.warning(
            "Prediction service is offline. Start the FastAPI backend to "
            "request a prediction."
        )

    st.divider()
    st.subheader("Get started")
    home_col, predict_col, info_col = st.columns(3)

    with home_col:
        st.markdown("#### Patient assessment")
        st.write("Enter dialysis and patient parameters to request a prediction.")

    with predict_col:
        st.markdown("#### Model details")
        st.write("Review the model type, required inputs, and decision threshold.")

    with info_col:
        st.markdown("#### Safe use")
        st.write("Learn how this prototype should—and should not—be used.")

    st.info(
        "This academic prototype is not a clinically validated diagnostic "
        "tool and should not guide treatment decisions."
    )


def render_prediction():
    st.title("Patient assessment")
    st.write(
        "Enter the patient and dialysis parameters below, then submit the form "
        "to request an IDH risk prediction."
    )

    if check_backend():
        st.caption("Prediction service: connected")
    else:
        st.warning("Cannot connect to the prediction service at 127.0.0.1:8000.")

    with st.form("prediction_form"):
        patient_col, treatment_col = st.columns(2)

        with patient_col:
            st.subheader("Patient information")
            D_Sexo = st.number_input("Sex (D_Sexo)", value=0.0)
            D_Edad = st.number_input("Age (D_Edad)", value=0.0)
            D_Dializador = st.number_input("Dialyzer (D_Dializador)", value=0.0)
            D_Bano = st.number_input("Dialysis bath (D_Bano)", value=0.0)
            D_Tecnica = st.number_input("Technique (D_Tecnica)", value=0.0)

            st.subheader("Weight information")
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
            st.subheader("Dialysis settings")
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
    st.write(", ".join(model_info["features"]))

    st.subheader("Categorical inputs")
    st.write(", ".join(model_info["categorical_features"]))

    st.caption(
        "The deployed API may apply its own threshold; the prediction result "
        "shows the threshold returned by the backend."
    )


def render_about():
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
    st.warning(
        "This is an academic prototype, not a clinically validated diagnostic "
        "tool. Predictions are informational only and must not replace "
        "professional clinical judgment."
    )


st.sidebar.title("IDH Prediction System")
page_names = ["Home", "Predict IDH", "Model Information", "About"]
selected_page = st.sidebar.radio(
    "Navigate",
    page_names,
    key="current_page"
)

if selected_page == "Home":
    render_home()
elif selected_page == "Predict IDH":
    render_prediction()
elif selected_page == "Model Information":
    render_model_information()
else:
    render_about()

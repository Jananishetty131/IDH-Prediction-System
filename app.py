
import streamlit as st
import requests

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="IDH Prediction System",
    page_icon="🏥",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000"

st.title("🏥 Intradialytic Hypotension Prediction")
st.write(
    "Enter patient dialysis parameters to predict "
    "the risk of intradialytic hypotension (IDH)."
)

# ---------------------------------
# BACKEND CONNECTION
# ---------------------------------

def check_backend():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        if response.status_code == 200:
            st.success("Backend connected successfully.")
            return True

        st.error("Backend is not responding correctly.")
        return False

    except requests.exceptions.RequestException:
        st.error(
            "Cannot connect to backend. "
            "Please start the FastAPI server."
        )
        return False


backend_connected = check_backend()

st.divider()

# ---------------------------------
# PATIENT INPUT FORM
# ---------------------------------

st.subheader("Patient Information")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Patient Details")

        sexo = st.number_input("Sex (D_Sexo)", value=0.0)
        edad = st.number_input("Age (D_Edad)", value=0.0)
        dializador = st.number_input("Dialyzer (D_Dializador)", value=0.0)
        bano = st.number_input("Dialysis Bath (D_Bano)", value=0.0)
        tecnica = st.number_input("Technique (D_Tecnica)", value=0.0)

        st.markdown("#### Weight Information")

        peso_seco = st.number_input("Dry Weight (D_PesoSeco)", value=0.0)
        peso_pre = st.number_input("Pre-dialysis Weight (D_PesoPre)", value=0.0)
        ganancia = st.number_input("Weight Gain (D_Ganancia)", value=0.0)

    with col2:
        D_Ganancia = st.number_input(
            "Weight Gain",
            min_value=0.0,
            value=2.0
        )

        D_TemperaturaBano = st.number_input(
            "Dialysate Temperature",
            min_value=0.0,
            value=36.0
        )

        D_FlujoSangre = st.number_input(
            "Blood Flow",
            min_value=0.0,
            value=300.0
        )

        D_FlujoBano = st.number_input(
            "Dialysate Flow",
            min_value=0.0,
            value=500.0
        )

        D_ConductividadBano = st.number_input(
            "Dialysate Conductivity",
            min_value=0.0,
            value=14.0
        )

        D_ConductividadBicarbonato = st.number_input(
            "Bicarbonate Conductivity",
            min_value=0.0,
            value=3.0
        )

    submit = st.form_submit_button(
        "Predict IDH",
        use_container_width=True
    )

# ---------------------------------
# PREDICTION
# ---------------------------------

if submit:

    if not backend_connected:
        st.error("Please start the backend before predicting.")

    else:

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
            "D_ConductividadBicarbonato": float(
                D_ConductividadBicarbonato
            )
        }

        try:
            with st.spinner("Predicting IDH..."):

                response = requests.post(
                    f"{API_URL}/predict",
                    json=patient_data,
                    timeout=30
                )

            if response.status_code == 200:

                result = response.json()

                probability = result["idh_probability"]
                prediction = result["prediction"]
                threshold = result.get("threshold", 0.35)

                st.divider()
                st.subheader("Prediction Result")

                st.metric(
                    "IDH Probability",
                    f"{probability * 100:.2f}%"
                )

                st.metric(
                    "Decision Threshold",
                    f"{threshold * 100:.0f}%"
                )

                if prediction == 1:
                    st.error(
                        "⚠️ IDH Present — Model predicts "
                        "a risk of intradialytic hypotension."
                    )
                else:
                    st.success(
                        "✅ IDH Not Predicted — Model does "
                        "not predict IDH for these inputs."
                    )

                st.caption(
                    "This is an academic prediction prototype, "
                    "not a clinically validated diagnostic tool."
                )

            else:
                st.error(
                    f"Prediction failed: {response.text}"
                )

        except requests.exceptions.RequestException as e:
            st.error(f"Backend connection error: {e}")

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="IDH Prediction System",
    page_icon="🏥",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "idh_pre_dialysis_model.joblib"

FEATURES = [
    "D_Sexo",
    "D_Edad",
    "D_Dializador",
    "D_Bano",
    "D_Tecnica",
    "D_PesoSeco",
    "D_PesoPre",
    "D_Ganancia",
    "D_TemperaturaBano",
    "D_FlujoSangre",
    "D_FlujoBano",
    "D_ConductividadBano",
    "D_ConductividadBicarbonato"
]
CATEGORICAL_FEATURES = [
    "D_Sexo",
    "D_Dializador",
    "D_Bano",
    "D_Tecnica"
]
THRESHOLD = 0.65


# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# -------------------------------
# CUSTOM STYLING
# -------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #F4F8FC;
}

.main-title {
    color: #174A75;
    font-size: 34px;
    font-weight: 700;
}

.subtitle {
    color: #607D96;
    font-size: 16px;
}

div.stButton > button {
    background-color: #1769AA;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #104D80;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# HEADER
# -------------------------------

st.markdown(
    '<div class="main-title">🏥 IDH Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intradialytic Hypotension Risk Prediction</div>',
    unsafe_allow_html=True
)

st.divider()

st.info(
    "Enter the patient's pre-dialysis information "
    "to generate an experimental IDH prediction."
)


# -------------------------------
# INPUT FORM
# -------------------------------

st.subheader("👤 Patient Information")

with st.form("idh_prediction_form"):

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
        st.markdown("#### Dialysis Parameters")

        temperatura = st.number_input(
            "Bath Temperature (D_TemperaturaBano)", value=0.0
        )

        flujo_sangre = st.number_input(
            "Blood Flow (D_FlujoSangre)", value=0.0
        )

        flujo_bano = st.number_input(
            "Dialysate Flow (D_FlujoBano)", value=0.0
        )

        conductividad = st.number_input(
            "Bath Conductivity (D_ConductividadBano)", value=0.0
        )

        bicarbonato = st.number_input(
            "Bicarbonate Conductivity (D_ConductividadBicarbonato)",
            value=0.0
        )

    submitted = st.form_submit_button("Predict IDH Risk")


# -------------------------------
# PREDICTION
# -------------------------------

if submitted:

    try:
        model = load_model()

        patient_data = {
            "D_Sexo": str(int(sexo)),
            "D_Edad": edad,
            "D_Dializador": str(int(dializador)),
            "D_Bano": str(int(bano)),
            "D_Tecnica": str(int(tecnica)),
            "D_PesoSeco": peso_seco,
            "D_PesoPre": peso_pre,
            "D_Ganancia": ganancia,
            "D_TemperaturaBano": temperatura,
            "D_FlujoSangre": flujo_sangre,
            "D_FlujoBano": flujo_bano,
            "D_ConductividadBano": conductividad,
            "D_ConductividadBicarbonato": bicarbonato
        }

        input_df = pd.DataFrame([patient_data], columns=FEATURES)

        probability = model.predict_proba(input_df)[0][1]

        prediction = int(probability >= THRESHOLD)

        st.divider()
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "IDH Probability",
                f"{probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Decision Threshold",
                f"{THRESHOLD * 100:.0f}%"
            )

        if prediction == 1:
            st.error("⚠️ Predicted Result: IDH Present")
        else:
            st.success("✅ Predicted Result: IDH Absent")

        st.progress(float(probability))

        st.caption(
            "Educational prototype only. This prediction is not "
            "a medical diagnosis or a substitute for clinical judgment."
        )

    except FileNotFoundError:
        st.error(
            "Model file not found. Check that "
            "idh_pre_dialysis_model.joblib is beside app.py."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")


# -------------------------------
# BATCH CSV PREDICTION
# -------------------------------

st.divider()
st.subheader("Batch Prediction")
st.caption("Upload a CSV containing the model's 13 input columns.")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    key="batch_prediction_csv"
)

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file, sep=None, engine="python")
        batch_df.columns = batch_df.columns.astype(str).str.strip()
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
    else:
        missing_features = [
            feature for feature in FEATURES if feature not in batch_df.columns
        ]

        if missing_features:
            st.error(
                "CSV is missing required columns: "
                + ", ".join(missing_features)
            )
        elif batch_df.empty:
            st.error("The uploaded CSV contains no rows.")
        else:
            input_df = batch_df.loc[:, FEATURES].copy()
            for feature in CATEGORICAL_FEATURES:
                input_df[feature] = input_df[feature].astype("string").str.strip()

            numeric_features = [
                feature for feature in FEATURES
                if feature not in CATEGORICAL_FEATURES
            ]
            for feature in numeric_features:
                input_df[feature] = pd.to_numeric(
                    input_df[feature], errors="coerce"
                )

            invalid_columns = input_df.columns[input_df.isna().any()].tolist()
            if invalid_columns:
                st.error(
                    "Missing or non-numeric values were found in: "
                    + ", ".join(invalid_columns)
                    + ". Correct these cells and upload the CSV again."
                )
            elif st.button("Predict uploaded rows", key="run_batch_prediction"):
                try:
                    model = load_model()
                    probabilities = model.predict_proba(input_df)[:, 1]
                    results_df = batch_df.copy()
                    results_df["IDH_Probability"] = probabilities
                    results_df["Predicted_IDH"] = [
                        "SI" if probability >= THRESHOLD else "NO"
                        for probability in probabilities
                    ]

                    st.success(f"Predictions complete for {len(results_df)} rows.")
                    st.dataframe(
                        results_df[["IDH_Probability", "Predicted_IDH"]].head(100),
                        use_container_width=True
                    )
                    st.download_button(
                        "Download prediction results",
                        data=results_df.to_csv(
                            index=False, sep=";"
                        ).encode("utf-8-sig"),
                        file_name="idh_prediction_results.csv",
                        mime="text/csv"
                    )
                    st.caption(
                        "Educational prototype only. Predictions are not a "
                        "medical diagnosis or a substitute for clinical judgment."
                    )
                except Exception as e:
                    st.error(f"Batch prediction error: {e}")


# -------------------------------
# FOOTER
# -------------------------------

st.divider()

st.caption(
    "IDH Prediction System | Machine Learning Based Academic Project"
)
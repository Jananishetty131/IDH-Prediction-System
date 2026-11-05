import json
import hashlib
import io
import math
import re
import shutil
import unicodedata
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
TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
SEX_CODES = {"Male": 0, "Female": 1}
PREDICTION_FIELDS = [
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
REPORT_FIELD_LABELS = {
    "D_Sexo": ("sex", "sexo", "gender", "d sexo"),
    "D_Edad": ("age", "edad", "d edad"),
    "D_Dializador": ("dialyzer", "dializador", "d dializador"),
    "D_Bano": ("dialysis bath", "bath", "bano", "baño", "d bano", "d baño"),
    "D_Tecnica": ("technique", "tecnica", "técnica", "d tecnica", "d técnica"),
    "D_PesoSeco": ("dry weight", "peso seco", "d pesoseco"),
    "D_PesoPre": (
        "pre dialysis weight", "pre-dialysis weight", "predialysis weight",
        "pre dialysis", "pre-dialysis",
        "peso pre", "peso predialisis", "d pesopre"
    ),
    "D_Ganancia": ("weight gain", "ganancia", "d ganancia"),
    "D_TemperaturaBano": (
        "dialysate temperature", "bath temperature", "temperatura bano",
        "temperatura-bano",
        "temperatura baño", "d temperaturabano"
    ),
    "D_FlujoSangre": ("blood flow", "flujo sangre", "d flujosangre"),
    "D_FlujoBano": (
        "dialysate flow", "bath flow", "flujo bano", "flujo baño", "d flujobano"
    ),
    "D_ConductividadBano": (
        "dialysate conductivity", "bath conductivity", "conductividad bano",
        "conductividad baño", "d conductividadbano"
    ),
    "D_ConductividadBicarbonato": (
        "bicarbonate conductivity", "conductividad bicarbonato",
        "d conductividadbicarbonato"
    )
}

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


def normalize_report_text(value):
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    )
    normalized = normalized.lower().replace("_", " ")
    return re.sub(r"[^a-z0-9.,+-]+", " ", normalized).strip()


def extract_report_parameters(text, word_data=None):
    aliases = []
    for field, labels in REPORT_FIELD_LABELS.items():
        for label in labels:
            normalized_label = normalize_report_text(label)
            if normalized_label:
                aliases.append((field, normalized_label))
    aliases.sort(key=lambda item: len(item[1]), reverse=True)

    lines = []
    if word_data:
        words = []
        for index, value in enumerate(word_data.get("text", [])):
            word = str(value).strip()
            if not word:
                continue
            try:
                left = int(word_data["left"][index])
                top = int(word_data["top"][index])
                height = int(word_data["height"][index])
            except (KeyError, TypeError, ValueError):
                continue
            words.append((top + height / 2, left, height, word))

        if words:
            typical_height = sorted(word[2] for word in words)[len(words) // 2]
            row_tolerance = max(5, min(typical_height * 0.65, 18))
            rows = []
            for center_y, left, _, word in sorted(words):
                matching_row = next(
                    (
                        row for row in rows
                        if abs(center_y - row["center"]) <= row_tolerance
                    ),
                    None
                )
                if matching_row is None:
                    rows.append({"center": center_y, "words": [(left, word)]})
                else:
                    word_count = len(matching_row["words"])
                    matching_row["center"] = (
                        matching_row["center"] * word_count + center_y
                    ) / (word_count + 1)
                    matching_row["words"].append((left, word))

            lines.extend(
                " ".join(word for _, word in sorted(row["words"]))
                for row in sorted(rows, key=lambda row: row["center"])
            )

    lines.extend(text.splitlines())
    extracted = {}
    for line in lines:
        normalized_line = normalize_report_text(line)
        matches = []
        for field, alias in aliases:
            match = re.search(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                normalized_line
            )
            if match and not any(
                match.start() < end and match.end() > start
                for start, end, _ in matches
            ):
                matches.append((match.start(), match.end(), field))

        matches.sort()
        for index, (_, end, field) in enumerate(matches):
            if field in extracted:
                continue
            next_start = (
                matches[index + 1][0]
                if index + 1 < len(matches)
                else len(normalized_line)
            )
            value_text = normalized_line[end:next_start]

            if field == "D_Sexo":
                if re.search(r"\bmale\b", value_text):
                    extracted[field] = "Male"
                    continue
                if re.search(r"\bfemale\b", value_text):
                    extracted[field] = "Female"
                    continue

            number_match = re.search(r"[-+]?\d+(?:[.,]\d+)?", value_text)
            if number_match:
                extracted[field] = number_match.group().replace(",", ".")

    return extracted


def render_prediction_result(patient_data):
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


def render_report_upload():
    st.markdown(
        '<div class="section-eyebrow">Local image extraction</div>',
        unsafe_allow_html=True
    )
    st.title("Upload a dialysis report")
    st.write(
        "Upload a JPG or JPEG report. OCR runs in the Streamlit server process "
        "in memory; this app does not send the image to an OCR provider or "
        "write it to disk. If hosted remotely, the image is sent to that host."
    )
    st.info(
        "OCR requires the Tesseract executable to be installed on this "
        "computer, in addition to the Python packages in requirements.txt."
    )

    uploaded_report = st.file_uploader(
        "Choose a report image",
        type=["jpg", "jpeg"],
        key="report_image_upload"
    )
    if uploaded_report is None:
        return

    if uploaded_report.size > 10 * 1024 * 1024:
        st.error("Please upload an image no larger than 10 MB.")
        return

    st.image(uploaded_report, caption="Selected report image", use_container_width=True)

    report_bytes = uploaded_report.getvalue()
    report_signature = hashlib.sha256(report_bytes).hexdigest()
    is_new_report = (
        report_signature != st.session_state.get("processed_report_signature")
    )
    retry_extraction = (
        not is_new_report
        and not st.session_state.get("report_image_ready")
        and st.button("Retry local extraction", use_container_width=True)
    )

    if is_new_report or retry_extraction:
        st.session_state["processed_report_signature"] = report_signature
        st.session_state["report_image_ready"] = False
        try:
            from PIL import Image, ImageOps, UnidentifiedImageError
            import pytesseract

            if TESSERACT_PATH.is_file():
                pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)
            elif shutil.which("tesseract") is None:
                st.error(
                    "Tesseract OCR was not found. Install it or add its "
                    "executable to PATH, then restart the app."
                )
                return

            with Image.open(io.BytesIO(report_bytes)) as source_image:
                image = ImageOps.exif_transpose(source_image).convert("RGB")
            with st.spinner("Reading report locally..."):
                report_text = pytesseract.image_to_string(image, lang="eng")
                word_data = pytesseract.image_to_data(
                    image,
                    lang="eng",
                    output_type=pytesseract.Output.DICT
                )
        except ModuleNotFoundError as error:
            st.error(
                f"Required Python package is missing: {error.name}. "
                "Install the project's requirements first."
            )
            return
        except pytesseract.TesseractNotFoundError:
            st.error(
                "Tesseract OCR was not found. Install the Tesseract executable "
                "and ensure it is available on PATH, then restart the app."
            )
            return
        except pytesseract.TesseractError as error:
            st.error(f"Local OCR could not read this image: {error}")
            return
        except (UnidentifiedImageError, OSError) as error:
            st.error(f"Could not open this report image: {error}")
            return

        extracted = extract_report_parameters(report_text, word_data)
        st.session_state["extraction_text"] = report_text
        st.session_state["extracted_parameters"] = extracted
        st.session_state["report_image_ready"] = True
        for field in PREDICTION_FIELDS:
            st.session_state[f"review_{field}"] = extracted.get(field, "")

        st.session_state["pending_page"] = "☑   Review & predict"
        st.rerun()


def render_report_review():
    st.markdown(
        '<div class="section-eyebrow">Verify before prediction</div>',
        unsafe_allow_html=True
    )
    st.title("Review extracted parameters")
    if not st.session_state.get("report_image_ready"):
        st.info("Upload a report image and run local extraction first.")
        return

    extracted = st.session_state.get("extracted_parameters", {})
    missing_fields = [
        field for field in PREDICTION_FIELDS
        if field not in extracted
    ]
    if missing_fields:
        st.warning(
            "OCR did not identify every required field. Fill in the blank "
            "values and compare all entries with the original report."
        )
    else:
        st.info(
            "OCR suggestions are not verified medical data. Check every value "
            "against the original report before confirming."
        )

    with st.expander("View locally extracted text"):
        st.text(st.session_state.get("extraction_text", ""))

    with st.form("report_review_form"):
        patient_col, treatment_col = st.columns(2)
        with patient_col:
            st.markdown("### Patient profile")
            detected_sex = extracted.get("D_Sexo", "")
            sex_index = (
                ["Select...", "Male", "Female"].index(detected_sex)
                if detected_sex in SEX_CODES
                else 0
            )
            selected_sex = st.selectbox(
                "Sex",
                ["Select...", *SEX_CODES.keys()],
                index=sex_index,
                key="review_sex"
            )
            st.caption(
                "Mapping is assumed: Male = 0, Female = 1. Confirm this matches "
                "the model's training encoding."
            )
            st.text_input(
                "Age (D_Edad)",
                key="review_D_Edad",
                placeholder="Enter age from report"
            )
            st.text_input(
                "Dialyzer (D_Dializador)",
                key="review_D_Dializador",
                placeholder="Enter dialyzer code from report"
            )
            st.text_input(
                "Dialysis bath (D_Bano)",
                key="review_D_Bano",
                placeholder="Enter bath code from report"
            )
            st.text_input(
                "Technique (D_Tecnica)",
                key="review_D_Tecnica",
                placeholder="Enter technique value from report"
            )
            st.markdown("### Weight information")
            st.text_input(
                "Dry weight (D_PesoSeco)",
                key="review_D_PesoSeco",
                placeholder="Enter dry weight from report"
            )
            st.text_input(
                "Pre-dialysis weight (D_PesoPre)",
                key="review_D_PesoPre",
                placeholder="Enter pre-dialysis weight from report"
            )
            st.text_input(
                "Weight gain (D_Ganancia)",
                key="review_D_Ganancia",
                placeholder="Enter weight gain from report"
            )

        with treatment_col:
            st.markdown("### Dialysis settings")
            st.text_input(
                "Dialysate temperature (D_TemperaturaBano)",
                key="review_D_TemperaturaBano",
                placeholder="Enter temperature from report"
            )
            st.text_input(
                "Blood flow (D_FlujoSangre)",
                key="review_D_FlujoSangre",
                placeholder="Enter blood flow from report"
            )
            st.text_input(
                "Dialysate flow (D_FlujoBano)",
                key="review_D_FlujoBano",
                placeholder="Enter dialysate flow from report"
            )
            st.text_input(
                "Dialysate conductivity (D_ConductividadBano)",
                key="review_D_ConductividadBano",
                placeholder="Enter conductivity from report"
            )
            st.text_input(
                "Bicarbonate conductivity (D_ConductividadBicarbonato)",
                key="review_D_ConductividadBicarbonato",
                placeholder="Enter bicarbonate conductivity from report"
            )

        confirmed = st.checkbox(
            "I checked every parameter against the report and confirm these "
            "values are correct.",
            key="review_confirmed"
        )
        predict = st.form_submit_button(
            "Confirm parameters and predict",
            use_container_width=True
        )

    if not predict:
        return
    if not confirmed:
        st.error("Review and confirm all parameters before predicting.")
        return
    if selected_sex == "Select...":
        st.error("Select Male or Female before predicting.")
        return

    patient_data = {"D_Sexo": SEX_CODES[selected_sex]}
    integer_fields = {"D_Dializador", "D_Bano", "D_Tecnica"}
    invalid_fields = []
    for field in PREDICTION_FIELDS:
        if field == "D_Sexo":
            continue
        raw_value = st.session_state.get(f"review_{field}", "").strip()
        if not raw_value:
            invalid_fields.append(field)
            continue
        try:
            numeric_value = float(raw_value.replace(",", "."))
            if not math.isfinite(numeric_value):
                raise ValueError
            if field in integer_fields:
                if not numeric_value.is_integer():
                    raise ValueError
                patient_data[field] = int(numeric_value)
            else:
                patient_data[field] = numeric_value
        except ValueError:
            invalid_fields.append(field)

    if invalid_fields:
        st.error(
            "Enter a valid number for: " + ", ".join(invalid_fields)
        )
        return

    render_prediction_result(patient_data)


def render_home():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Dialysis care · Risk assessment</div>
            <div class="hero-title">IDH Prediction System</div>
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
        "1. Open **Patient assessment** for manual entry, or **Report image** "
        "to extract values from a JPG/JPEG.\n"
        "2. For image reports, check every OCR suggestion on **Review & "
        "predict** and correct missing or incorrect values.\n"
        "3. Confirm the values to send them to the local prediction API.\n"
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
    "▧   Report image": "Report image",
    "☑   Review & predict": "Review & predict",
    "▤   Model information": "Model Information",
    "ⓘ   About": "About"
}
page_labels = list(page_names)
pending_page = st.session_state.pop("pending_page", None)
if pending_page in page_labels:
    st.session_state["current_page"] = pending_page

selected_page = st.sidebar.radio(
    "WORKSPACE",
    page_labels,
    key="current_page"
)

if page_names[selected_page] == "Home":
    render_home()
elif page_names[selected_page] == "Predict IDH":
    render_prediction()
elif page_names[selected_page] == "Report image":
    render_report_upload()
elif page_names[selected_page] == "Review & predict":
    render_report_review()
elif page_names[selected_page] == "Model Information":
    render_model_information()
else:
    render_about()

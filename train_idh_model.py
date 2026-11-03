
from pathlib import Path
import zipfile
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "New_IDH_Dataset" / "dataset_11.csv.zip"

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {ZIP_PATH}")

# --------------------------------------------------
# 2. LOAD DATASET 11
# --------------------------------------------------

with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    csv_files = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv")
    ]

    if not csv_files:
        raise ValueError("No CSV file found inside ZIP.")

    with archive.open(csv_files[0]) as file:
        df = pd.read_csv(
            file,
            sep=";",
            index_col=False,
            usecols=range(25)
        )

df.columns = df.columns.str.strip()

print("\nDATASET LOADED")
print("Shape:", df.shape)

# --------------------------------------------------
# 3. PREPARE TARGET
# --------------------------------------------------

df["D_Hipotension"] = (
    df["D_Hipotension"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df = df[df["D_Hipotension"].isin(["NO", "SI"])].copy()

df["IDH"] = df["D_Hipotension"].map({
    "NO": 0,
    "SI": 1
})

print("\nTARGET DISTRIBUTION:")
print(df["IDH"].value_counts())

# --------------------------------------------------
# 4. SELECT PRE-DIALYSIS FEATURES
# --------------------------------------------------

# Exclude direct blood-pressure outcome fields,
# post-dialysis weight, and variables not needed
# for this initial pre-dialysis baseline.

features = [
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

missing = [col for col in features if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

X = df[features].copy()
y = df["IDH"].copy()

categorical_features = [
    "D_Sexo",
    "D_Dializador",
    "D_Bano",
    "D_Tecnica"
]

numerical_features = [
    col for col in features
    if col not in categorical_features
]

for col in categorical_features:
    X[col] = X[col].astype("string")

for col in numerical_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# --------------------------------------------------
# 5. SPLIT DATA
# --------------------------------------------------

# 70% training, 15% validation, 15% testing.

X_train, X_remaining, y_train, y_remaining = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.50,
    random_state=42,
    stratify=y_remaining
)

print("\nDATA SPLIT:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# --------------------------------------------------
# 6. PREPROCESSING
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# --------------------------------------------------
# 7. TRAIN MULTIPLE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

trained_models = {}
validation_scores = {}

print("\nTRAINING MODELS...")

for name, classifier in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    val_probabilities = pipeline.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, val_probabilities)

    trained_models[name] = pipeline
    validation_scores[name] = auc

    print(f"{name}: Validation ROC-AUC = {auc:.4f}")

# --------------------------------------------------
# 8. SELECT BEST MODEL USING VALIDATION DATA
# --------------------------------------------------

best_name = max(validation_scores, key=validation_scores.get)
best_model = trained_models[best_name]

print("\nBEST MODEL:", best_name)

# --------------------------------------------------
# 9. SELECT A SENSITIVITY-FOCUSED THRESHOLD
# --------------------------------------------------

val_probabilities = best_model.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in np.arange(0.05, 0.96, 0.01):

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        val_predictions,
        labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    if sensitivity >= 0.80:
        threshold_results.append(
            (threshold, sensitivity, specificity)
        )

if threshold_results:
    # Among thresholds achieving at least 80% sensitivity,
    # choose the one with the highest specificity.
    threshold, _, _ = max(
        threshold_results,
        key=lambda result: (result[2], result[0])
    )
else:
    # If 80% sensitivity cannot be reached, choose
    # the threshold with the highest validation sensitivity.
    threshold = max(
        np.arange(0.05, 0.96, 0.01),
        key=lambda t: recall_score(
            y_val,
            (val_probabilities >= t).astype(int)
        )
    )

print("Selected threshold:", round(float(threshold), 2))

# --------------------------------------------------
# 10. FINAL TEST EVALUATION
# --------------------------------------------------

test_probabilities = best_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

sensitivity = tp / (tp + fn) if (tp + fn) else 0
specificity = tn / (tn + fp) if (tn + fp) else 0

print("\n========== FINAL TEST RESULTS ==========")

print("Model:", best_name)
print("Accuracy:", round(accuracy_score(y_test, test_predictions), 4))
print("Precision:", round(precision_score(y_test, test_predictions, zero_division=0), 4))
print("Recall / Sensitivity:", round(sensitivity, 4))
print("Specificity:", round(specificity, 4))
print("F1 Score:", round(f1_score(y_test, test_predictions, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, test_probabilities), 4))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, test_predictions))

print("\nTN:", tn, "FP:", fp)
print("FN:", fn, "TP:", tp)

# --------------------------------------------------
# 11. SAVE MODEL AND INFORMATION
# --------------------------------------------------

model_path = BASE_DIR / "idh_pre_dialysis_model.joblib"
info_path = BASE_DIR / "idh_model_info.json"

joblib.dump(best_model, model_path)

model_info = {
    "model_name": best_name,
    "features": features,
    "categorical_features": categorical_features,
    "threshold": float(threshold),
    "target_mapping": {
        "NO": 0,
        "SI": 1
    },
    "dataset": "Dataset 11",
    "random_state": 42
}

with open(info_path, "w") as file:
    json.dump(model_info, file, indent=4)

print("\nMODEL SAVED SUCCESSFULLY!")
print("Model:", model_path)
print("Information:", info_path)

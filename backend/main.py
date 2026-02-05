from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

app = FastAPI()

# ================= CORS =================
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:5501",
    "http://127.0.0.1:8000",
    "https://rop-ai-2.onrender.com",
    "https://kalimeriem.github.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= LOAD MODEL =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "xgboost_model.pkl")
encoder_path = os.path.join(BASE_DIR, "model", "encoder.pkl")
features_path = os.path.join(BASE_DIR, "model", "features.pkl")

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)
features = joblib.load(features_path)

print("✅ Model + encoder + features loaded")

# ================= COLUMN TYPES =================
categorical_cols = [
    "bit_model",
    "bit_type",
    "bi_serial_number",
    "Manufactor",
    "formation",
    "bit_category_final",
    "bit_technology"
]

categorical_int_cols = ["from_month", "from_hour"]

# ================= HOME =================
@app.get("/")
def home():
    return {"message": "ROP prediction API running"}

# ================= PREDICT =================
@app.post("/predict")
def predict(data: dict):
    try:
        # ===== Convert to DataFrame =====
        df = pd.DataFrame([data])

        # ===== Ensure categorical as string =====
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        for col in categorical_int_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # ===== ENCODE EXACTLY LIKE TRAINING =====
        cat_cols_present = [
            c for c in categorical_cols + categorical_int_cols if c in df.columns
        ]

        encoded_df = pd.DataFrame()

        if cat_cols_present:
            encoded = encoder.transform(df[cat_cols_present])
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out()
            )

        # remove categorical original
        df_numeric = df.drop(columns=cat_cols_present, errors="ignore").reset_index(drop=True)

        # combine numeric + encoded
        df = pd.concat([df_numeric, encoded_df], axis=1)

        # ===== ADD DEFAULT MISSING FIELDS =====
        default_zero_cols = [
            "wob_max",
            "WELL_Name_id",
            "location_id"
        ]

        for col in default_zero_cols:
            if col not in df.columns:
                df[col] = 0

        # ===== ADD ANY MISSING TRAINING COLUMNS =====
        for col in features:
            if col not in df.columns:
                df[col] = 0

        # ===== ORDER SAME AS TRAINING =====
        df = df[features]

        # ===== DEBUG (shows in render logs) =====
        print("Incoming columns:", len(df.columns))
        print("Expected columns:", len(features))

        # ===== PREDICT =====
        pred = model.predict(df)[0]
        pred = max(pred, 0)

        return {"ROP_prediction": float(pred)}

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}

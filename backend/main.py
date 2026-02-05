from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

# ===== CORS SETTINGS =====
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:5501",
    "http://127.0.0.1:8000",
    "https://kalimeriem.github.io",
    "https://rop-ai-1.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== LOAD MODEL FILES =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "xgboost_model.pkl")
encoder_path = os.path.join(BASE_DIR, "model", "encoder.pkl")
features_path = os.path.join(BASE_DIR, "model", "features.pkl")

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)
features = joblib.load(features_path)



print("Model + encoder loaded successfully")

# ===== INPUT COLUMN TYPES =====
categorical_cols = [
    'bit_model',
    'bit_type',
    'bi_serial_number',
    'Manufactor',
    'formation',
    'bit_category_final',
    'bit_technology'
]
categorical_int_cols = ['from_month', 'from_hour']

# ===== HOME =====
@app.get("/")
def home():
    return {"message": "ROP prediction API running"}

# ===== PREDICT =====
@app.post("/predict")
def predict(data: dict):

    try:
        df = pd.DataFrame([data])

        # ===== ensure categorical as string =====
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        for col in categorical_int_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # ===== ENCODE (CRITICAL FIX HERE) =====
        cat_cols = [c for c in categorical_cols + categorical_int_cols if c in df.columns]

        if cat_cols:
            encoded = encoder.transform(df[cat_cols])

          
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out()
            )

            df = df.drop(columns=cat_cols).reset_index(drop=True)
            df = pd.concat([df, encoded_df], axis=1)

        # ===== ADD ANY MISSING TRAINING FEATURES =====
        for col in features:
            if col not in df.columns:
                df[col] = 0

        # ===== ORDER SAME AS TRAINING =====
        df = df[features]

        print("Incoming shape:", df.shape)  # debug in render logs

        # ===== PREDICT =====
        pred = model.predict(df)[0]
        pred = max(pred, 0)

        return {"ROP_prediction": float(pred)}

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}


# main.py — Expense Forecasting API
# Jalankan lokal  : uvicorn main:app --reload --port 8000
# Deploy Railway  : otomatis via Procfile

import os
import io
import pickle
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional


# PATH MODEL — relatif terhadap lokasi main.py
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "saved_model", "expense_forecasting_model.keras")
SCALER_PATH   = os.path.join(BASE_DIR, "saved_model", "scaler.pkl")
USER_MAP_PATH = os.path.join(BASE_DIR, "saved_model", "user_to_idx.pkl")

class LinearScaleLayer(layers.Layer):
    """Layer tipis di output model (scale + bias trainable)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale", shape=(1,),
            initializer="ones", trainable=True
        )
        self.bias = self.add_weight(
            name="bias", shape=(1,),
            initializer="zeros", trainable=True
        )

    def call(self, x):
        return x * self.scale + self.bias

    def get_config(self):
        return super().get_config()


class HuberLoss(keras.losses.Loss):
    """Custom Huber loss yang dipakai saat training."""
    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta  = delta
        self._huber = keras.losses.Huber(delta=delta)

    def call(self, y_true, y_pred):
        return self._huber(y_true, y_pred)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"delta": self.delta})
        return cfg


# LOAD MODEL, SCALER, USER MAPPING
print("=" * 50)
print("Loading model artifacts...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "LinearScaleLayer": LinearScaleLayer,
        "HuberLoss":        HuberLoss,
    }
)
print("✅ Model loaded.")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
print("✅ Scaler (RobustScaler) loaded.")

with open(USER_MAP_PATH, "rb") as f:
    user_to_idx = pickle.load(f)
print(f"✅ User mapping loaded — {len(user_to_idx)} users dikenal model.")
print("=" * 50)


# KONSTANTA
FEATURE_COLS = [
    "expense_log",
    "txn_count",
    "unique_categories",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_std_3",
    "month_sin",
    "month_cos",
]
LOOK_BACK = 12


# FEATURE ENGINEERING
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"]       = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    monthly_expense = (
        df[df["type"] == "expense"]
        .groupby("year_month")["amount"]
        .sum()
        .reset_index(name="monthly_expense")
    )
    txn_count = (
        df.groupby("year_month")
        .size()
        .reset_index(name="txn_count")
    )
    unique_cat = (
        df.groupby("year_month")["category"]
        .nunique()
        .reset_index(name="unique_categories")
    )

    monthly = monthly_expense.merge(txn_count,  on="year_month", how="left")
    monthly = monthly.merge(unique_cat,          on="year_month", how="left")
    monthly = monthly.sort_values("year_month").reset_index(drop=True)

    monthly["expense_log"]    = np.log1p(monthly["monthly_expense"])
    monthly["lag_1"]          = monthly["expense_log"].shift(1).fillna(0)
    monthly["lag_2"]          = monthly["expense_log"].shift(2).fillna(0)
    monthly["lag_3"]          = monthly["expense_log"].shift(3).fillna(0)
    monthly["rolling_mean_3"] = monthly["expense_log"].rolling(3).mean().fillna(0)
    monthly["rolling_std_3"]  = monthly["expense_log"].rolling(3).std().fillna(0)
    monthly["month"]          = monthly["year_month"].dt.month
    monthly["month_sin"]      = np.sin(2 * np.pi * monthly["month"] / 12)
    monthly["month_cos"]      = np.cos(2 * np.pi * monthly["month"] / 12)

    monthly = monthly.fillna(0)   # ⚠️ wajib — konsisten dengan training
    return monthly


# PREDICTION PIPELINE
def run_prediction(df: pd.DataFrame, user_id: str) -> dict:
    if user_id not in user_to_idx:
        raise ValueError(
            f"user_id '{user_id}' tidak dikenal oleh model."
        )

    monthly = build_features(df)

    if len(monthly) < LOOK_BACK:
        raise ValueError(
            f"Data kurang. Butuh minimal {LOOK_BACK} bulan, "
            f"tapi hanya ada {len(monthly)} bulan."
        )

    features        = monthly[FEATURE_COLS].values
    features_scaled = scaler.transform(features)
    X_seq  = features_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, len(FEATURE_COLS))
    X_user = np.array([[user_to_idx[user_id]]])

    pred_scaled = model.predict([X_seq, X_user], verbose=0)

    dummy       = np.zeros((1, len(FEATURE_COLS)))
    dummy[0, 0] = pred_scaled[0][0]
    pred_log    = scaler.inverse_transform(dummy)[0][0]
    pred_rupiah = float(np.expm1(pred_log))

    today = datetime.date.today()
    if today.month == 12:
        next_month = today.replace(year=today.year + 1, month=1, day=1)
    else:
        next_month = today.replace(month=today.month + 1, day=1)

    return {
        "user_id":                     user_id,
        "predicted_expense":           round(pred_rupiah, 2),
        "predicted_expense_formatted": f"Rp {pred_rupiah:,.0f}",
        "prediction_for_month":        next_month.strftime("%B %Y"),
        "data_months_used":            len(monthly),
        "status":                      "success",
    }


# FASTAPI APP
app = FastAPI(
    title="Expense Forecasting API",
    description="Prediksi pengeluaran bulanan menggunakan GRU + Attention.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Transaction(BaseModel):
    user_id:        str
    date:           str
    amount:         float
    type:           str
    category:       str
    payment_method: Optional[str] = "unknown"

class JSONPredictRequest(BaseModel):
    user_id:      str
    transactions: List[Transaction]


@app.get("/", summary="Health Check")
def root():
    return {
        "status":    "running",
        "message":   "Expense Forecasting API aktif",
        "timestamp": datetime.datetime.now().isoformat(),
    }


@app.post("/predict/csv", summary="Prediksi via Upload CSV")
async def predict_csv(
    file:    UploadFile = File(...),
    user_id: str        = Form(...),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File harus berformat .csv")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca CSV: {e}")

    required_cols = {"user_id", "date", "amount", "type", "category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Kolom tidak ada di CSV: {missing}")

    df_user = df[df["user_id"] == user_id].copy()
    if df_user.empty:
        raise HTTPException(status_code=404, detail=f"user_id '{user_id}' tidak ditemukan.")

    try:
        return run_prediction(df_user, user_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediksi: {e}")


@app.post("/predict/json", summary="Prediksi via JSON")
def predict_json(body: JSONPredictRequest):
    df = pd.DataFrame([t.dict() for t in body.transactions])
    try:
        return run_prediction(df, body.user_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediksi: {e}")
# main.py — Expense Forecasting API (Per-User Model)
# Jalankan lokal  : uvicorn main:app --reload --port 8000
# Deploy Railway  : otomatis via Procfile

import os
import io
import pickle
import datetime
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# KONSTANTA
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR     = os.path.join(BASE_DIR, "saved_model")
LOOK_BACK    = 3
FEATURE_COLS = [
    "expense_log", "txn_count", "unique_categories",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3", "rolling_std_3",
    "month_sin", "month_cos",
]
TRAIN_THRESHOLD = 3   # min bulan agar user baru di-autotrain

os.makedirs(SAVE_DIR, exist_ok=True)


# CUSTOM LAYERS & LOSS 
@tf.keras.utils.register_keras_serializable()
class LinearScaleLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name="scale", shape=(1,), initializer="ones",  trainable=True)
        self.bias  = self.add_weight(name="bias",  shape=(1,), initializer="zeros", trainable=True)

    def call(self, x):
        return x * self.scale + self.bias

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class HuberLoss(keras.losses.Loss):
    def __init__(self, delta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta  = delta
        self._huber = keras.losses.Huber(delta=delta)

    def call(self, y_true, y_pred):
        return self._huber(y_true, y_pred)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"delta": self.delta})
        return cfg


# FEATURE ENGINEERING 
def build_monthly_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Buat monthly aggregation dari raw transaction DataFrame."""
    df = df.copy()
    df["date"]       = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    monthly_expense = (
        df[df["type"] == "expense"]
        .groupby("year_month")["amount"].sum()
        .reset_index(name="monthly_expense")
    )
    txn_count  = df.groupby("year_month").size().reset_index(name="txn_count")
    unique_cat = df.groupby("year_month")["category"].nunique().reset_index(name="unique_categories")

    monthly = monthly_expense.merge(txn_count,  on="year_month", how="left")
    monthly = monthly.merge(unique_cat,          on="year_month", how="left")
    monthly = monthly.fillna(0).sort_values("year_month").reset_index(drop=True)
    return monthly


def build_features(user_monthly: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering dari monthly DataFrame."""
    df = user_monthly.copy().sort_values("year_month").reset_index(drop=True)

    df["expense_log"]    = np.log1p(df["monthly_expense"])
    df["lag_1"]          = df["expense_log"].shift(1)
    df["lag_2"]          = df["expense_log"].shift(2)
    df["lag_3"]          = df["expense_log"].shift(3)
    df["rolling_mean_3"] = df["expense_log"].rolling(3).mean()
    df["rolling_std_3"]  = df["expense_log"].rolling(3).std()
    df["month"]          = pd.to_datetime(df["year_month"]).dt.month
    df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)
    df = df.fillna(0)
    return df


# MODEL BUILDER 
def build_gru_model(n_features: int, look_back: int) -> keras.Model:
    seq_input = keras.Input(shape=(look_back, n_features), name="sequence_input")

    x    = layers.GRU(64, return_sequences=True)(seq_input)
    x    = layers.Dropout(0.2)(x)
    x    = layers.GRU(32, return_sequences=True)(x)
    x    = layers.Dropout(0.2)(x)
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x    = layers.LayerNormalization()(x + attn)
    x    = layers.GlobalAveragePooling1D()(x)
    x    = layers.Dense(32, activation="relu")(x)
    x    = layers.Dropout(0.1)(x)
    x    = layers.Dense(1)(x)
    out  = LinearScaleLayer(name="linear_scale")(x)

    model = keras.Model(inputs=seq_input, outputs=out, name="GRU_Attention_PerUser")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss=HuberLoss(delta=1.0),
        metrics=["mae"],
    )
    return model


# TRAINING 
def train_user_model(user_id: str, user_monthly: pd.DataFrame, verbose: int = 0) -> dict:
    user_dir = os.path.join(SAVE_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    df     = build_features(user_monthly)
    scaler = RobustScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    values  = df[FEATURE_COLS].values
    targets = df["expense_log"].values

    effective_look_back = min(LOOK_BACK, max(1, len(values) - 2))
    X, y = [], []
    for i in range(effective_look_back, len(values)):
        X.append(values[i - effective_look_back:i])
        y.append(targets[i])

    X, y = np.array(X), np.array(y)
    if len(X) == 0:
        return {"user_id": user_id, "mae": None, "rmse": None, "status": "not_enough_data"}

    use_val_split = len(X) >= 5
    split         = int(len(X) * 0.8) if use_val_split else len(X)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_gru_model(n_features=len(FEATURE_COLS), look_back=effective_look_back)

    callbacks = [
        EarlyStopping(monitor="val_loss" if use_val_split else "loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss" if use_val_split else "loss", factor=0.5, patience=5, verbose=0),
    ]
    model.fit(
        X_train, y_train,
        validation_split=0.2 if use_val_split else 0.0,
        epochs=100,
        batch_size=min(16, len(X_train)),
        callbacks=callbacks,
        verbose=verbose,
    )

    mae, rmse = 0.0, 0.0
    if len(X_test) > 0:
        pred_log = model.predict(X_test, verbose=0).flatten()
        dummy      = np.zeros((len(pred_log), len(FEATURE_COLS)));  dummy[:, 0] = pred_log
        pred_real  = np.expm1(scaler.inverse_transform(dummy)[:, 0])
        dummy2     = np.zeros((len(y_test), len(FEATURE_COLS)));    dummy2[:, 0] = y_test
        true_real  = np.expm1(scaler.inverse_transform(dummy2)[:, 0])
        mae  = float(mean_absolute_error(true_real, pred_real))
        rmse = float(np.sqrt(mean_squared_error(true_real, pred_real)))

    model.save(os.path.join(user_dir, "model.keras"))
    with open(os.path.join(user_dir, "scaler.pkl"),    "wb") as f: pickle.dump(scaler, f)
    with open(os.path.join(user_dir, "look_back.pkl"), "wb") as f: pickle.dump(effective_look_back, f)

    return {"user_id": user_id, "mae": mae, "rmse": rmse, "status": "trained"}


# PREDICTORS 
class PersonalModelPredictor:
    def __init__(self, user_id: str):
        user_dir = os.path.join(SAVE_DIR, user_id)
        self.model = tf.keras.models.load_model(
            os.path.join(user_dir, "model.keras"),
            custom_objects={"LinearScaleLayer": LinearScaleLayer, "HuberLoss": HuberLoss},
        )
        with open(os.path.join(user_dir, "scaler.pkl"),    "rb") as f: self.scaler    = pickle.load(f)
        with open(os.path.join(user_dir, "look_back.pkl"), "rb") as f: self.look_back = pickle.load(f)

    def predict(self, user_monthly: pd.DataFrame) -> tuple:
        df     = build_features(user_monthly)
        values = df[FEATURE_COLS].values

        if len(values) < self.look_back:
            pad    = np.zeros((self.look_back - len(values), len(FEATURE_COLS)))
            values = np.vstack([pad, values])

        seq_scaled  = self.scaler.transform(values[-self.look_back:])
        X           = np.expand_dims(seq_scaled, axis=0)
        pred_scaled = self.model.predict(X, verbose=0).flatten()

        dummy       = np.zeros((1, len(FEATURE_COLS)));  dummy[0, 0] = pred_scaled[0]
        pred_real   = float(np.expm1(self.scaler.inverse_transform(dummy)[0, 0]))
        confidence  = "high" if len(user_monthly) >= 12 else "medium"
        return max(pred_real, 0.0), confidence


class ColdStartPredictor:
    def predict(
        self,
        onboarding_estimate: Optional[float],
        months_of_data: int,
        actual_expenses: Optional[list] = None,
    ) -> tuple:
        if months_of_data == 0:
            if onboarding_estimate and onboarding_estimate > 0:
                return max(onboarding_estimate, 0.0), "low"
            raise ValueError("User baru tanpa data aktual harus mengisi onboarding_estimate.")

        actual_mean = float(np.mean(actual_expenses))
        if onboarding_estimate and onboarding_estimate > 0:
            w_actual    = min(0.5 + months_of_data * 0.15, 1.0)
            prediction  = w_actual * actual_mean + (1 - w_actual) * onboarding_estimate
        else:
            prediction  = actual_mean

        confidence = "medium" if months_of_data >= 2 else "low"
        return max(prediction, 0.0), confidence


class HybridPredictor:
    def __init__(self):
        self.cold_start = ColdStartPredictor()
        self._cache: dict = {}
        self._lock  = threading.Lock()

    def _has_model(self, user_id: str) -> bool:
        return os.path.exists(os.path.join(SAVE_DIR, user_id, "model.keras"))

    def _load(self, user_id: str) -> PersonalModelPredictor:
        with self._lock:
            if user_id not in self._cache:
                self._cache[user_id] = PersonalModelPredictor(user_id)
        return self._cache[user_id]

    def _invalidate(self, user_id: str):
        with self._lock:
            self._cache.pop(user_id, None)

    def predict(
        self,
        user_id: str,
        user_monthly: Optional[pd.DataFrame] = None,
        onboarding_estimate: Optional[float] = None,
    ) -> dict:
        months = len(user_monthly) if user_monthly is not None else 0

        # Punya model personal
        if self._has_model(user_id):
            predictor      = self._load(user_id)
            df_hist        = user_monthly if user_monthly is not None else pd.DataFrame(columns=["year_month", "monthly_expense", "txn_count", "unique_categories"])
            predicted, conf = predictor.predict(df_hist)
            mode  = "personal_model"
            note  = f"Model personal {user_id} ({months} bulan data)"

        # User baru, data cukup → auto-train
        elif months >= TRAIN_THRESHOLD:
            train_user_model(user_id, user_monthly, verbose=0)
            self._invalidate(user_id)
            predictor       = self._load(user_id)
            predicted, conf = predictor.predict(user_monthly)
            mode  = "personal_model"
            note  = f"Model personal baru auto-trained ({months} bulan data)"

        # Cold start
        else:
            actual = user_monthly["monthly_expense"].tolist() if months > 0 else None
            predicted, conf = self.cold_start.predict(
                onboarding_estimate=onboarding_estimate,
                months_of_data=months,
                actual_expenses=actual,
            )
            mode = "cold_start"
            src  = []
            if onboarding_estimate: src.append("estimasi onboarding")
            if months > 0:          src.append(f"{months} bulan data aktual")
            note = "Cold start: " + (", ".join(src) if src else "tidak ada data")

        today = datetime.date.today()
        next_month = today.replace(year=today.year + 1, month=1, day=1) if today.month == 12 \
                     else today.replace(month=today.month + 1, day=1)

        return {
            "user_id":                     user_id,
            "predicted_expense":           round(predicted, 2),
            "predicted_expense_formatted": f"Rp {predicted:,.0f}",
            "prediction_for_month":        next_month.strftime("%B %Y"),
            "mode":                        mode,
            "confidence":                  conf,
            "months_of_data":              months,
            "note":                        note,
            "status":                      "success",
        }


# Inisialisasi predictor global
predictor = HybridPredictor()


# FASTAPI APP 
app = FastAPI(
    title="Expense Forecasting API — Per User",
    description="Prediksi pengeluaran bulanan dengan model personal per user (GRU + Attention). "
                "Mendukung personal model, auto-train, dan cold start.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# REQUEST / RESPONSE SCHEMAS 
class Transaction(BaseModel):
    date:           str
    amount:         float
    type:           str           # "expense" | "income"
    category:       str
    subcategory:    Optional[str] = None
    payment_method: Optional[str] = "unknown"
    description:    Optional[str] = None


class JSONPredictRequest(BaseModel):
    user_id:             str
    transactions:        Optional[List[Transaction]] = None
    onboarding_estimate: Optional[float] = None    # untuk user baru tanpa data


class TrainRequest(BaseModel):
    user_id:      str
    transactions: List[Transaction]


# HELPER 
def transactions_to_monthly(user_id: str, transactions: List[Transaction]) -> pd.DataFrame:
    """Konversi list Transaction → monthly aggregation DataFrame."""
    rows = [{"user_id": user_id, **t.dict()} for t in transactions]
    df   = pd.DataFrame(rows)
    return build_monthly_from_raw(df)


# ENDPOINTS 
@app.get("/", summary="Health Check")
def root():
    return {
        "status":    "running",
        "message":   "Expense Forecasting API (Per-User) aktif",
        "version":   "3.0.0",
        "timestamp": datetime.datetime.now().isoformat(),
    }


@app.get("/users", summary="List User yang Sudah Punya Model")
def list_users():
    """Mengembalikan daftar user_id yang sudah punya model personal tersimpan."""
    users = []
    if os.path.isdir(SAVE_DIR):
        for name in sorted(os.listdir(SAVE_DIR)):
            path = os.path.join(SAVE_DIR, name, "model.keras")
            if os.path.exists(path):
                users.append(name)
    return {"users": users, "total": len(users)}


@app.post("/predict/json", summary="Prediksi via JSON (Hybrid)")
def predict_json(body: JSONPredictRequest):
    """
    Prediksi pengeluaran bulan depan.

    - **user_id** yang sudah punya model personal → pakai model tersebut.
    - **user_id** baru dengan data ≥ 3 bulan → auto-train lalu prediksi.
    - **user_id** baru dengan data < 3 bulan → cold start (butuh `onboarding_estimate`).
    """
    try:
        user_monthly = None
        if body.transactions:
            user_monthly = transactions_to_monthly(body.user_id, body.transactions)

        return predictor.predict(
            user_id=body.user_id,
            user_monthly=user_monthly,
            onboarding_estimate=body.onboarding_estimate,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediksi: {e}")


@app.post("/predict/csv", summary="Prediksi via Upload CSV")
async def predict_csv(
    file:                UploadFile    = File(...),
    user_id:             str           = Form(...),
    onboarding_estimate: Optional[float] = Form(None),
):
    """
    Upload file CSV transaksi. Kolom wajib: `date`, `amount`, `type`, `category`.
    Kolom `user_id` di dalam CSV akan difilter otomatis berdasarkan `user_id` di form.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File harus berformat .csv")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca CSV: {e}")

    required = {"date", "amount", "type", "category"}
    missing  = required - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Kolom tidak ada di CSV: {missing}")

    # Filter per user jika kolom user_id ada
    if "user_id" in df.columns:
        df = df[df["user_id"] == user_id].copy()
        if df.empty:
            raise HTTPException(status_code=404, detail=f"user_id '{user_id}' tidak ditemukan di CSV.")
    else:
        df["user_id"] = user_id

    try:
        user_monthly = build_monthly_from_raw(df)
        return predictor.predict(
            user_id=user_id,
            user_monthly=user_monthly,
            onboarding_estimate=onboarding_estimate,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediksi: {e}")


@app.post("/train", summary="Train / Re-train Model User Tertentu")
def train(body: TrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger training (atau re-training) model personal untuk satu user.
    Training dijalankan di background — endpoint langsung return `accepted`.
    Gunakan `GET /users` untuk cek apakah model sudah tersedia.
    """
    if len(body.transactions) < TRAIN_THRESHOLD:
        raise HTTPException(
            status_code=422,
            detail=f"Butuh minimal {TRAIN_THRESHOLD} bulan data untuk training. "
                   f"Saat ini hanya {len(body.transactions)} transaksi."
        )

    def _train_bg():
        user_monthly = transactions_to_monthly(body.user_id, body.transactions)
        train_user_model(body.user_id, user_monthly, verbose=0)
        predictor._invalidate(body.user_id)

    background_tasks.add_task(_train_bg)

    return {
        "status":  "accepted",
        "message": f"Training model untuk '{body.user_id}' dimulai di background.",
        "user_id": body.user_id,
    }


@app.delete("/model/{user_id}", summary="Hapus Model User")
def delete_model(user_id: str):
    """Hapus model personal user (misalnya untuk reset / re-train dari nol)."""
    import shutil
    user_dir = os.path.join(SAVE_DIR, user_id)
    if not os.path.isdir(user_dir):
        raise HTTPException(status_code=404, detail=f"Model untuk '{user_id}' tidak ditemukan.")
    shutil.rmtree(user_dir)
    predictor._invalidate(user_id)
    return {"status": "deleted", "user_id": user_id}

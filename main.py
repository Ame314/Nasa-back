from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# 🚀 Inicializar FastAPI
app = FastAPI(title="Clasificador de Exoplanetas 🚀")

# 🌍 Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📂 Cargar modelo y codificador
rf = joblib.load("model/rf_exoplanets.pkl")
enc = joblib.load("model/label_encoder.pkl")

# 📑 Columnas esperadas
feature_cols = [
    "koi_period", "koi_duration", "koi_depth", "koi_impact",
    "koi_prad", "koi_teq", "koi_insol",
    "koi_steff", "koi_srad", "koi_slogg", "koi_kepmag",
    "koi_model_snr", "koi_fpflag_nt", "koi_fpflag_ss",
    "koi_fpflag_co", "koi_fpflag_ec"
]

# 🛰️ Esquema de entrada
class ExoplanetFeatures(BaseModel):
    koi_period: float | None = None
    koi_duration: float | None = None
    koi_depth: float | None = None
    koi_impact: float | None = None
    koi_prad: float | None = None
    koi_teq: float | None = None
    koi_insol: float | None = None
    koi_steff: float | None = None
    koi_srad: float | None = None
    koi_slogg: float | None = None
    koi_kepmag: float | None = None
    koi_model_snr: float | None = None
    koi_fpflag_nt: int = 0
    koi_fpflag_ss: int = 0
    koi_fpflag_co: int = 0
    koi_fpflag_ec: int = 0


@app.get("/")
def root():
    return {"msg": "Bienvenido al clasificador de exoplanetas 🚀"}


@app.post("/predict")
def predict(features: ExoplanetFeatures):
    data = features.dict()
    X = pd.DataFrame([[data[c] for c in feature_cols]], columns=feature_cols)
    X = X.fillna(X.median(numeric_only=True))

    proba = rf.predict_proba(X)[0]
    pred = enc.classes_[int(np.argmax(proba))]

    return {
        "prediccion": pred,
        "probabilidades": {enc.classes_[i]: float(proba[i]) for i in range(len(proba))}
    }


@app.post("/analisis")
def analisis(features: ExoplanetFeatures):
    data = features.dict()

    # 📌 1. Clasificación del modelo
    X = pd.DataFrame([[data[c] for c in feature_cols]], columns=feature_cols)
    X = X.fillna(X.median(numeric_only=True))
    proba = rf.predict_proba(X)[0]
    pred = enc.classes_[int(np.argmax(proba))]

    # 📌 2. Habitabilidad simple
    habitabilidad = []
    if data["koi_prad"] and 0.5 <= data["koi_prad"] <= 2.0:
        habitabilidad.append("Radio compatible con planeta rocoso 🌍")
    else:
        habitabilidad.append("Radio sugiere gigante gaseoso o planeta enano 🪐")

    if data["koi_teq"] and 180 <= data["koi_teq"] <= 310:
        habitabilidad.append("Temperatura en zona habitable 🧊🔥")
    else:
        habitabilidad.append("Temperatura fuera de zona habitable ❌")

    if data["koi_insol"] and 0.25 <= data["koi_insol"] <= 2:
        habitabilidad.append("Flujo de radiación adecuado ☀️")
    else:
        habitabilidad.append("Flujo de radiación extremo ❌")

    # 📌 3. Estrella
    estrella = {
        "Temperatura (K)": data.get("koi_steff"),
        "Radio (R☉)": data.get("koi_srad"),
        "Magnitud Kepler": data.get("koi_kepmag"),
    }

    # 📌 4. Fiabilidad
    flags = {
        "Not Transit-Like": data.get("koi_fpflag_nt", 0),
        "Stellar Eclipse": data.get("koi_fpflag_ss", 0),
        "Centroid Offset": data.get("koi_fpflag_co", 0),
        "Ephemeris Contamination": data.get("koi_fpflag_ec", 0),
    }
    fiabilidad = "Posible falso positivo 🚨" if any(flags.values()) else "Candidato fiable ✅"

    return {
        "clasificacion_modelo": pred,
        "probabilidades": {enc.classes_[i]: float(proba[i]) for i in range(len(proba))},
        "habitabilidad": habitabilidad,
        "estrella": estrella,
        "fiabilidad": fiabilidad
    }

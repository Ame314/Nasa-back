
Paso 1) 
crear la carpeta y crear un entorno virtual:
- python -m venv venv
- venv\Scripts\activate
- 
*Instalar los requerimientos*
`
pip install -r requirements.txt

fastapi
uvicorn
pandas
numpy
joblib
scikit-learn
`
![[Pasted image 20251003231943.png]]
lo que se puede ver son el modelo que determina si es un exoplaneta y los parámetros que se utilizan para determinar esto
es decir lo que hace que funcione
lo siguiente es armar la api, esto se hace con fast api para hacer lo rápido y las pruebas se hacen de una con localhost:#puerto/docs
acá se pueden hacer pruebas con los endpoints que fueron creados

este es el contenido base del main.py que llama al ML para que diga que es y una cosa mas de datos curiosos que podríamos poner para que sea más chevere 

### **1. Importar bibliotecas**



```
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
```

- **FastAPI**: Framework para crear APIs web rápidas y eficientes.
- **CORSMiddleware**: Permite que el servidor acepte solicitudes desde cualquier origen (como un navegador).
- **pydantic**: Se usa para definir modelos de datos (estructuras de entrada).
- **pandas, numpy**: Para manipular datos y cálculos numéricos.
- **joblib**: Para cargar modelos de machine learning previamente entrenados.

---

### 🚀 **2. Inicializar FastAPI**



```
app = FastAPI(title="Clasificador de Exoplanetas 🚀")
```

- Se crea una aplicación FastAPI con el título "Clasificador de Exoplanetas".

---

### 🌍 **3. Habilitar CORS**


```
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- Esto permite que el API acepte solicitudes desde cualquier dominio (por ejemplo, desde un frontend en otro puerto o sitio web).

---

### 📂 **4. Cargar el modelo y el codificador**



```
rf = joblib.load("model/rf_exoplanets.pkl")
enc = joblib.load("model/label_encoder.pkl")
```

- **rf**: Es un modelo de Random Forest entrenado para clasificar exoplanetas.
- **enc**: Es un codificador que transforma las etiquetas de clasificación (como "exoplaneta", "no exoplaneta") en números.

---

### 📑 **5. Columnas esperadas en la entrada**



```
feature_cols = [
    "koi_period", "koi_duration", "koi_depth", "koi_impact",
    "koi_prad", "koi_teq", "koi_insol",
    "koi_steff", "koi_srad", "koi_slogg", "koi_kepmag",
    "koi_model_snr", "koi_fpflag_nt", "koi_fpflag_ss",
    "koi_fpflag_co", "koi_fpflag_ec"
]
```

- Estas son las columnas de datos que el modelo espera recibir para hacer una predicción.

---

### 🛰️ **6. Esquema de entrada (pydantic)**



```
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
```

- Define el formato de los datos que se enviarán al servidor para hacer una predicción.
- Cada atributo representa una característica del exoplaneta o de la estrella.

---

### 📦 **7. Ruta raíz**


```
@app.get("/")
def root():
    return {"msg": "Bienvenido al clasificador de exoplanetas 🚀"}
```

- Esta ruta es para probar que el servidor está funcionando. Si visitas `http://localhost:8000/`, te mostrará un mensaje de bienvenida.

---

### 🧠 **8. Ruta para hacer una predicción**


```
@app.post("/predict")
def predict(features: ExoplanetFeatures):
    data = features.dict()
    X = pd.DataFrame([[data[c] for c in feature_cols]], columns=feature_cols)
    X = X.fillna(X.median(numeric_only=True))

    proba = rf.predict_proba(X)
    pred = enc.classes_[int(np.argmax(proba))]

    return {
        "prediccion": pred,
        "probabilidades": {enc.classes_[i]: float(proba[i]) for i in range(len(proba))}
    }
```

- **features**: Es el objeto que contiene los datos de entrada.
- **X**: Se convierte en un DataFrame de Pandas para que el modelo lo entienda.
- **fillna**: Llena los valores faltantes con la mediana de los datos numéricos.
- **proba**: Contiene las probabilidades de que el objeto sea de cada clase.
- **pred**: Es la predicción final del modelo.

---

### 🧪 **9. Ruta para análisis detallado**





```
@app.post("/analisis")
def analisis(features: ExoplanetFeatures):
    data = features.dict()

    # 📌 1. Clasificación del modelo
    X = pd.DataFrame([[data[c] for c in feature_cols]], columns=feature_cols)
    X = X.fillna(X.median(numeric_only=True))
    proba = rf.predict_proba(X)
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
        "Cent
```

```
1. Importar bibliotecas

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
FastAPI: Framework para crear APIs web rápidas y eficientes.
CORSMiddleware: Permite que el servidor acepte solicitudes desde cualquier origen (como un navegador).
pydantic: Se usa para definir modelos de datos (estructuras de entrada).
pandas, numpy: Para manipular datos y cálculos numéricos.
joblib: Para cargar modelos de machine learning previamente entrenados.
🚀 2. Inicializar FastAPI

app = FastAPI(title="Clasificador de Exoplanetas 🚀")
Se crea una aplicación FastAPI con el título "Clasificador de Exoplanetas".
🌍 3. Habilitar CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Esto permite que el API acepte solicitudes desde cualquier dominio (por ejemplo, desde un frontend en otro puerto o sitio web).
📂 4. Cargar el modelo y el codificador

rf = joblib.load("model/rf_exoplanets.pkl")
enc = joblib.load("model/label_encoder.pkl")
rf: Es un modelo de Random Forest entrenado para clasificar exoplanetas.
enc: Es un codificador que transforma las etiquetas de clasificación (como "exoplaneta", "no exoplaneta") en números.
📑 5. Columnas esperadas en la entrada

feature_cols = [
    "koi_period", "koi_duration", "koi_depth", "koi_impact",
    "koi_prad", "koi_teq", "koi_insol",
    "koi_steff", "koi_srad", "koi_slogg", "koi_kepmag",
    "koi_model_snr", "koi_fpflag_nt", "koi_fpflag_ss",
    "koi_fpflag_co", "koi_fpflag_ec"
]
Estas son las columnas de datos que el modelo espera recibir para hacer una predicción.
🛰️ 6. Esquema de entrada (pydantic)

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
Define el formato de los datos que se enviarán al servidor para hacer una predicción.
Cada atributo representa una característica del exoplaneta o de la estrella.
📦 7. Ruta raíz

@app.get("/")
def root():
    return {"msg": "Bienvenido al clasificador de exoplanetas 🚀"}
Esta ruta es para probar que el servidor está funcionando. Si visitas http://localhost:8000/, te mostrará un mensaje de bienvenida.
🧠 8. Ruta para hacer una predicción

@app.post("/predict")
def predict(features: ExoplanetFeatures):
    data = features.dict()
    X = pd.DataFrame([[data[c] for c in feature_cols]], columns=feature_cols)
    X = X.fillna(X.median(numeric_only=True))

    proba = rf.predict_proba(X)
    pred = enc.classes_[int(np.argmax(proba))]

    return {
        "prediccion": pred,
        "probabilidades": {enc.classes_[i]: float(proba[i]) for i in range(len(proba))}
    }
features: Es el objeto que contiene los datos de entrada.
X: Se convierte en un DataFrame de Pandas para que el modelo lo entienda.
fillna: Llena los valores faltantes con la mediana de los datos numéricos.
proba: Contiene las probabilidades de que el objeto sea de cada clase.
pred: Es la predicción final del modelo.
🧪 9. Ruta para análisis detallado

@app.post("/analisis")
def analisis(features: ExoplanetFeatures):
    data = features.dict()

    # 📌 1. Clasificación del modelo
    X = pd.DataFrame([[data[c] for c in feature_cols]], columns=feature_cols)
    X = X.fillna(X.median(numeric_only=True))
    proba = rf.predict_proba(X)
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
        "Cent
```

## iniciar el back 
`venv\Scripts\activate`
`uvicorn main:app --reload --port 8000`

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from starlette.middleware.cors import CORSMiddleware

app = FastAPI(title="API de Prédiction de Crédit")

origins = [
    "http://52.89.55.119",  # Remplacez par votre IP publique ou domaine
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:4200",  # Si vous testez en local avec Angular
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Autoriser uniquement ces origines
    allow_credentials=True,
    allow_methods=["*"],  # Autoriser toutes les méthodes HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Autoriser tous les headers
)
# Middleware CORS pour autoriser les requêtes de n'importe quel domaine
# Charger le modèle et le scaler depuis les fichiers pré-entraînés
model = joblib.load('best_xgb_model.pkl')
scaler = joblib.load('scaler.pkl')  # Charger le scaler déjà ajusté


# Modèle Pydantic pour les données du formulaire
class CreditData(BaseModel):
    num__EXT_SOURCE_2: float
    num__EXT_SOURCE_3: float
    num__DAYS_BIRTH: float
    num__DAYS_ID_PUBLISH: float
    num__DAYS_REGISTRATION: float
    num__SK_ID_CURR: float
    num__AMT_ANNUITY: float
    num__DAYS_LAST_PHONE_CHANGE: float
    num__DAYS_EMPLOYED: float
    num__AMT_CREDIT: float
    num__AMT_INCOME_TOTAL: float
    num__REGION_POPULATION_RELATIVE: float
    num__AMT_GOODS_PRICE: float
    num__HOUR_APPR_PROCESS_START: float
    num__TOTALAREA_MODE: float
    num__AMT_REQ_CREDIT_BUREAU_YEAR: float
    num__YEARS_BEGINEXPLUATATION_MEDI: float
    num__YEARS_BEGINEXPLUATATION_MODE: float
    num__YEARS_BEGINEXPLUATATION_AVG: float
    num__OBS_60_CNT_SOCIAL_CIRCLE: float


@app.post("/api/predict/")
async def predict(credit_data: CreditData):
    try:
        # Convertir l'objet Pydantic en dictionnaire, puis en DataFrame
        data_dict = credit_data.dict()
        df = pd.DataFrame([data_dict])
        print("Données reçues :\n", df.head())

        # Encodage des variables catégorielles (si des colonnes sont de type 'object')
        le = LabelEncoder()
        for col in df.select_dtypes(include="object").columns:
            df[col] = le.fit_transform(df[col])
        print("Encodages des variables catégorielles effectué.")

        # Sélectionner uniquement les colonnes numériques pour la mise à l'échelle
        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
        scaled_data = scaler.transform(df[numeric_columns])
        print("Données mises à l'échelle.")

        # Prédiction : obtenir la probabilité de la classe positive
        predictions_prob = model.predict_proba(scaled_data)[:, 1]
        # Pour cet exemple, nous considérons qu'il y a une seule prédiction
        prob = float(predictions_prob[0])

        # Déterminer la décision en fonction du seuil 0.5
        credit_decision = "Crédit Accordé" if prob > 0.05 else "Crédit Refusé"

        # Renvoi personnalisé : décision et probabilité de remboursement
        return {
            "credit_decision": credit_decision,
            "repayment_probability": prob
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

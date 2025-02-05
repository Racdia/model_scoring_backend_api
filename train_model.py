import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import pandas as pd

# Charger les données prétraitées
X_train = pd.read_pickle('X_train.pkl')
X_test = pd.read_pickle('X_test.pkl')
y_train = pd.read_pickle('y_train.pkl')
y_test = pd.read_pickle('y_test.pkl')

# Créer l'instance de SimpleImputer et l'appliquer aux données
imputer = SimpleImputer(strategy='mean')  # Imputation avec la moyenne
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)  # Utilisation du même imputeur sur le jeu de test

# Entraîner plusieurs modèles
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier()
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    f1 = f1_score(y_test, y_pred)
    results[model_name] = {
        'AUC-ROC': auc_roc,
        'F1-score': f1,
        'Classification Report': classification_report(y_test, y_pred)
    }

    # Sauvegarder le modèle entraîné
    with open(f'{model_name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Afficher les résultats
for model_name, result in results.items():
    print(f"{model_name} - AUC-ROC: {result['AUC-ROC']}")
    print(f"{model_name} - F1-score: {result['F1-score']}")
    print(result['Classification Report'])

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
import joblib
import torch
import lightgbm as lgb

app = FastAPI(title="Telco Churn Prediction API")

# ── Load preprocessors ──────────────────────────────────────────────────────
scaler   = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

with open("models/mlp_config.json") as f:
    mlp_cfg = json.load(f)

CAT_COLS = mlp_cfg["cat_cols"]
NUM_COLS = mlp_cfg["num_cols"]

# ── Load LightGBM ────────────────────────────────────────────────────────────
lgbm_model = lgb.Booster(model_file="models/tree_model.txt")

# ── Load MLP ─────────────────────────────────────────────────────────────────
# Import the model class — in production you'd package this properly
import sys, os
sys.path.append(os.path.dirname(__file__))

import torch.nn as nn

class TabularMLP(nn.Module):
    def __init__(self, cardinalities, n_numerical, hidden_sizes, dropout=0.3):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cats, min(50, (n_cats + 1) // 2))
            for n_cats in cardinalities
        ])
        emb_dim = sum(min(50, (n + 1) // 2) for n in cardinalities)
        input_dim = n_numerical + emb_dim
        layers = []
        for h in hidden_sizes:
            layers += [nn.Linear(input_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + embs, dim=1)
        return self.mlp(x).squeeze(1)

mlp_model = TabularMLP(
    cardinalities=mlp_cfg["cardinalities"],
    n_numerical=mlp_cfg["n_numerical"],
    hidden_sizes=mlp_cfg["hidden_sizes"],
    dropout=mlp_cfg["dropout"],
)
mlp_model.load_state_dict(torch.load("models/mlp_model.pth", map_location="cpu"))
mlp_model.eval()


# ── Input schema ─────────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


def preprocess(data: CustomerFeatures):
    import pandas as pd
    row = pd.DataFrame([data.dict()])
    row["SeniorCitizen"] = row["SeniorCitizen"].astype(str)

    # Encode categoricals
    for col in CAT_COLS:
        row[col] = encoders[col].transform(row[col])

    X_num = scaler.transform(row[NUM_COLS].values.astype(float))
    X_cat = row[CAT_COLS].values

    return X_num, X_cat


@app.get("/")
def root():
    return {"message": "Telco Churn API — POST to /predict/mlp or /predict/tree"}


@app.post("/predict/mlp")
def predict_mlp(data: CustomerFeatures):
    X_num, X_cat = preprocess(data)
    x_num_t = torch.tensor(X_num, dtype=torch.float32)
    x_cat_t = torch.tensor(X_cat, dtype=torch.long)
    with torch.no_grad():
        prob = torch.sigmoid(mlp_model(x_num_t, x_cat_t)).item()
    return {"prediction": "Yes" if prob >= 0.5 else "No", "probability": round(prob, 4)}


@app.post("/predict/tree")
def predict_tree(data: CustomerFeatures):
    X_num, X_cat = preprocess(data)
    X = np.hstack([X_num, X_cat])
    prob = float(lgbm_model.predict(X)[0])
    return {"prediction": "Yes" if prob >= 0.5 else "No", "probability": round(prob, 4)}

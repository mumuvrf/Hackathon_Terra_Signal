from fastapi import FastAPI
import pandas as pd
import pickle

from process import process

app = FastAPI()

# Carrega modelo e transformer
with open("../models/transformer.pkl", "rb") as f:
    transformer = pickle.load(f)

with open("../models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Carrega histórico para recomendação
history = pd.read_csv("../data/history.csv")
history["MonthlyCharges"] = pd.to_numeric(history["MonthlyCharges"], errors="coerce")
monthly_75 = history["MonthlyCharges"].quantile(0.75)


# -----------------------------
# 1) Endpoint de Predição
# -----------------------------
@app.post("/predict")
def predict(client: dict):

    # Converte entrada em DF
    df = pd.DataFrame([client])

    # Processa dados com a MESMA função usada no treino
    df_processed = process(path="../data/inference.csv")  # NÃO usar, então ajustamos:

    # Simplesmente aplicamos o process diretamente no df:
    df_processed = df.copy()
    df_processed = process_dataframe_like_process(df_processed)

    # Remove colunas como no treino
    drop_cols = ['TotalCharges', 'CustomerFeedback', 'customerID', 'tenure',
                  'MonthlyCharges', 'MonthlyIncome']

    for col in drop_cols:
        if col in df_processed.columns:
            df_processed = df_processed.drop(columns=[col])

    # Transforma
    X = transformer.transform(df_processed)

    # Prediz
    pred = model.predict(X)[0]
    churn = "Yes" if pred == 1 else "No"

    return {"Churn": churn}


# Função auxiliar super simples que replica o process() sem CSV
def process_dataframe_like_process(df):
    # Converte campos numéricos
    df["tenure"] = pd.to_numeric(df.get("tenure"), errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df.get("MonthlyCharges"), errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df.get("TotalCharges"), errors="coerce")

    # Regra de TotalCharges (igual à original)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])

    # Regra de tenure (igual à original)
    df["tenure"] = df["tenure"].fillna(
        (df["TotalCharges"] // df["MonthlyCharges"]).fillna(0)
    )

    df = df.dropna()

    # Categorias
    df["tenureCategory"] = df["tenure"].apply(
        lambda x: "0-11" if x < 12 else
                  "12-23" if x < 24 else
                  "24-35" if x < 36 else
                  "36-47" if x < 48 else "48+"
    )

    df["MonthlyChargesCategory"] = df["MonthlyCharges"].apply(
        lambda x: "0-40" if x <= 40 else
                  "41-80" if x <= 80 else
                  "80+"
    )

    return df


# -----------------------------
# 2) Endpoint de Recomendações
# -----------------------------
@app.post("/recommend")
def recommend(client: dict):

    recs = []

    # Regra 1: contrato mensal
    if client.get("Contract") == "Month-to-month":
        recs.append("Recomendar um plano de maior duração (1 ano ou 2 anos).")

    # Regra 2: top 25% em MonthlyCharges
    if float(client.get("MonthlyCharges", 0)) >= monthly_75:
        recs.append("Oferecer desconto (está no top 25% de MonthlyCharges).")

    # Regra 3: Fiber optic
    if client.get("InternetService") == "Fiber optic":
        recs.append("Recomendar migração para DSL.")

    return {"recommendations": recs}

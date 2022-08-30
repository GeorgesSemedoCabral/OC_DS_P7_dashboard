import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
lgbm_model = joblib.load("balanced_lgbm_model.sav")

class ClientID(BaseModel):
    SK_ID_CURR: int
    threshold: float

@app.post("/predict")
def profile_and_predict(client: ClientID):
    data = client.dict()
    client_df = pd.read_csv("../../input/application_test.csv")
    client_df.fillna("No information", inplace=True)
    app_client = client_df[client_df["SK_ID_CURR"]==data["SK_ID_CURR"]]
    gender = app_client.loc[:, "CODE_GENDER"].to_numpy().item(0)
    days_birth = app_client.loc[:, "DAYS_BIRTH"].to_numpy().item(0)
    age = int(np.abs(days_birth) / 365.25)
    status = app_client.loc[:, "NAME_FAMILY_STATUS"].to_numpy().item(0)
    children = app_client.loc[:, "CNT_CHILDREN"].to_numpy().item(0)
    housing = app_client.loc[:, "NAME_HOUSING_TYPE"].to_numpy().item(0)
    education = app_client.loc[:, "NAME_EDUCATION_TYPE"].to_numpy().item(0)
    occupation = app_client.loc[:, "OCCUPATION_TYPE"].to_numpy().item(0)
    organization = app_client.loc[:, "ORGANIZATION_TYPE"].to_numpy().item(0)
    income = app_client.loc[:, "AMT_INCOME_TOTAL"].to_numpy().item(0)
    income_euro = int(income / 73.3670)
    test_df = pd.read_csv("preprocessed_test_df.csv")
    feats = [f for f in test_df.columns if f not in [
        "TARGET",
        "SK_ID_BUREAU",
        "SK_ID_PREV",
        "index"
    ]]
    test_feats = test_df[feats]
    app_df = test_feats[test_feats["SK_ID_CURR"]==data["SK_ID_CURR"]]
    app_test = app_df.drop(columns="SK_ID_CURR")
    proba = lgbm_model.predict_proba(app_test).item(1)
    if proba >= data["threshold"]:
        score = "B"
    else:
        score = "G"
    return {
        "client_ID": data["SK_ID_CURR"],
        "gender": gender,
        "age": age,
        "status": status,
        "children": children,
        "housing": housing,
        "education": education,
        "occupation": occupation,
        "organization": organization,
        "monthly income (€, 05-18-2018)": income_euro,
        "PROBA": proba,
        "SCORE": score
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
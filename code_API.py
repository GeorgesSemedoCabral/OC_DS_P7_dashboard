import joblib
import pandas as pd
import shap
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
client_df = pd.read_csv("data/client_test.csv")
client_df.fillna("No information", inplace=True)
test_df = pd.read_csv("data/split_csv_pandas/chunk0.csv")
for i in range(1, 5):
    df_chunk = pd.read_csv("data/split_csv_pandas/chunk{}.csv".format(i))
    test_df = test_df.append(df_chunk)
    del df_chunk
feats = [f for f in test_df.columns if f not in [
    "TARGET",
    "SK_ID_BUREAU",
    "SK_ID_PREV",
    "index"
]]
test_feats = test_df[feats]
lgbm_model = joblib.load("models/balanced_lgbm_model.sav")
explainer = shap.TreeExplainer(lgbm_model)

class ClientID(BaseModel):
    SK_ID_CURR: int
    threshold: float

@app.post("/predict")
def profile_and_predict(client: ClientID):
    data = client.dict()
    app_client = client_df[client_df["SK_ID_CURR"]==data["SK_ID_CURR"]]
    gender = app_client.loc[:, "CODE_GENDER"].to_numpy().item(0)
    age = app_client.loc[:, "DAYS_BIRTH"].to_numpy().item(0)
    status = app_client.loc[:, "NAME_FAMILY_STATUS"].to_numpy().item(0)
    children = app_client.loc[:, "CNT_CHILDREN"].to_numpy().item(0)
    housing = app_client.loc[:, "NAME_HOUSING_TYPE"].to_numpy().item(0)
    education = app_client.loc[:, "NAME_EDUCATION_TYPE"].to_numpy().item(0)
    occupation = app_client.loc[:, "OCCUPATION_TYPE"].to_numpy().item(0)
    organization = app_client.loc[:, "ORGANIZATION_TYPE"].to_numpy().item(0)
    income = app_client.loc[:, "AMT_INCOME_TOTAL"].to_numpy().item(0)
    credit = app_client.loc[:, "AMT_CREDIT"].to_numpy().item(0)
    annuity = app_client.loc[:, "AMT_ANNUITY"].to_numpy().item(0)
    app_df = test_feats[test_feats["SK_ID_CURR"]==data["SK_ID_CURR"]]
    app_test = app_df.drop(columns="SK_ID_CURR")
    proba = lgbm_model.predict_proba(
        app_test,
        num_iteration=lgbm_model.best_iteration_
    ).item(1)
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
        "monthly income (€, 05-18-2018)": income,
        "credit (€)": credit,
        "annuities (€)": annuity,
        "PROBA": proba,
        "SCORE": score
    }

class ClientID2(BaseModel):
    SK_ID_CURR: int

@app.post("/features")
def client_features(client: ClientID2):
    data = client.dict()
    app_df = test_feats[test_feats["SK_ID_CURR"]==data["SK_ID_CURR"]]
    app_test = app_df.drop(columns="SK_ID_CURR")
    shap_values = explainer.shap_values(app_test)
    return {
        "explain_value": explainer.expected_value[1],
        "shap_values": shap_values[1].tolist(),
        "app_values": app_test.values.tolist(),
        "features": app_test.columns.tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app)
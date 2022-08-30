import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
lgbm_model = joblib.load("balanced_lgbm_model.sav")

class ClientID(BaseModel):
    SK_ID_CURR : int

@app.post("/predict")
def profile_and_predict(client: ClientID):
    data = client.dict()
    client_df = pd.read_csv("original_test_df.csv")
    app_client = client_df[client_df["SK_ID_CURR"]==data["SK_ID_CURR"]]
    if app_client.loc[:, "CODE_GENDER"].to_numpy().item(0) == 0:
        gender = "M"
    elif app_client.loc[:, "CODE_GENDER"].to_numpy().item(0) == 1:
        gender = "F"
    children = app_client.loc[:, "CNT_CHILDREN"].to_numpy().item(0)
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
    if lgbm_model.predict(app_test)[0] == 0:
        score = "G"
    elif lgbm_model.predict(app_test)[0] == 1:
        score = "B"
    proba = lgbm_model.predict_proba(app_test).item(1)
    return {
        "client_ID": data["SK_ID_CURR"],
        "gender": gender,
        "children": children,
        "SCORE": score,
        "PROBA": proba
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
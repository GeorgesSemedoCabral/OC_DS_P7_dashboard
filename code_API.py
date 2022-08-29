import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
lgbm_model = joblib.load("balanced_lgbm_model.sav")

class ClientIndex(BaseModel):
    client_index : int

@app.post("/predict")
def predict_scoring(client: ClientIndex):
    data = client.dict()
    df = pd.read_csv("preprocessed_test_df.csv")
    feats = [f for f in df.columns if f not in [
        "TARGET",
        "SK_ID_CURR",
        "SK_ID_BUREAU",
        "SK_ID_PREV",
        "index"
    ]]
    app_df = df[feats]
    app_client = app_df.iloc[[data["client_index"]]]
    score = lgbm_model.predict(app_client)[0]
    proba = lgbm_model.predict_proba(app_client).max()
    return {
        "score": score,
        "proba": proba
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
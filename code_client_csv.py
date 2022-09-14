import numpy as np
import pandas as pd

client = pd.read_csv("../../input/application_test.csv")
client.fillna("NaN", inplace=True)
client["DAYS_BIRTH"] = client["DAYS_BIRTH"].apply(
    lambda x: int(np.abs(x) / 365.25)
)
client["ORGANIZATION_TYPE"] = client["ORGANIZATION_TYPE"].apply(
    lambda x: x.split(":")[0]
)
client["ORGANIZATION_TYPE"] = client["ORGANIZATION_TYPE"].apply(
    lambda x: x.split(" Entity")[0]
)
client["AMT_INCOME_TOTAL"] = client["AMT_INCOME_TOTAL"].apply(
    lambda x: int(x / 73.3670)
)
client["AMT_ANNUITY"] = client["AMT_ANNUITY"].astype("float64")

client.to_csv("data/client_test.csv", index=False)
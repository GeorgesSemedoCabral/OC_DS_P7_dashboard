import joblib
import numpy as np
import pandas as pd

### Ce code est une annexe au dashboard. Il prétraite les données brutes de la base de données clients pour leur exploration dans le dashboard. A utiliser périodiquement (ex.: tous les jours à minuit) pour tenir compte des MàJ de la BDD.

client = pd.read_csv("../../input/application_test.csv")
client.fillna("NaN", inplace=True)

# Conversion de l'âge en années

client["DAYS_BIRTH"] = client["DAYS_BIRTH"].apply(
    lambda x: int(np.abs(x) / 365.25)
)
client.rename(columns={"DAYS_BIRTH": "YEARS_BIRTH"}, inplace=True)

# Découpage en classes d'âges

age_bins = [17, 24, 34, 44, 54, 64, 74]
age_labels = ["25-", "25-34", "35-44", "45-54", "55-64", "65+"]
client["YEARS_CLASS"] = pd.cut(
    client["YEARS_BIRTH"],
    age_bins,
    labels=age_labels,
    ordered=False
).astype(str)

# Découpage en classes de nombre d'enfants (notamment pour regrouper les familles dites "nombreuses")

child_bins = [-1, 0, 1, 2, 20]
child_labels = ["0", "1", "2", "3+"]
client["CHILD_CLASS"] = pd.cut(
    client["CNT_CHILDREN"],
    child_bins,
    labels=child_labels,
    ordered=False
).astype(str)

# Regroupement des types d'organisations professionnelles

client["ORGANIZATION_TYPE"] = client["ORGANIZATION_TYPE"].apply(
    lambda x: x.split(":")[0]
)
client["ORGANIZATION_TYPE"] = client["ORGANIZATION_TYPE"].apply(
    lambda x: x.split(" Entity")[0]
)

# Conversion des revenus mensuels et des montants des crédits en € (sur la base du taux de conversion rouble-euro à la date du 18 mai 2018)

client["AMT_INCOME_TOTAL"] = client["AMT_INCOME_TOTAL"].apply(
    lambda x: int(x / 73.3670)
)
client["AMT_CREDIT"] = client["AMT_CREDIT"].apply(
    lambda x: int(x / 73.3670)
)

joblib.dump(client, "data/client_test.sav")
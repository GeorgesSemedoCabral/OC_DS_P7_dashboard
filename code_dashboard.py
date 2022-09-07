import pandas as pd
#import plotly.express as px
import plotly.graph_objects as go
import requests
import shap
from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots

client = pd.read_csv("client_test.csv")

#client_ID = {"SK_ID_CURR": 100001, "threshold": 0.5}
#score = requests.post("http://127.0.0.1:8000/predict", json=client_ID)
#client_ID2 = {"SK_ID_CURR": 100001}
#features = requests.post("http://127.0.0.1:8000/features", json=client_ID2)

app = Dash(__name__)

app.layout = html.Div([
    html.Label("Analyse globale"),
    dcc.Graph(id="graph_01"),
    html.P("Names"),
    dcc.Dropdown(
        id="names_drop",
        options=[
            "CODE_GENDER",
            "DAYS_BIRTH",
            "NAME_FAMILY_STATUS",
            "CNT_CHILDREN",
            "NAME_HOUSING_TYPE",
            "NAME_EDUCATION_TYPE",
            "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE"
        ],
        value="CODE_GENDER",
        clearable=False
    ),
    html.P("Values"),
    dcc.Dropdown(
        id="values_drop",
        options=[
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY"
        ],
        value="AMT_INCOME_TOTAL",
        clearable=False
    ),
])

@app.callback(
    Output("graph_01", "figure"),
    Input("names_drop", "value"),
    Input("values_drop", "value"),
)
def generate_figure(names_drop, values_drop):
    #fig = go.Figure()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Names", "Values"],
        specs=[[{"type":"pie"}, {"type":"pie"}]]
    )
    fig.add_trace(
        go.Pie(
            labels=list(client[names_drop].unique()),
            values=list(client[names_drop].value_counts(sort=False)),
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Pie(
            labels=list(client[names_drop].unique()),
            values=list(
                client.groupby(names_drop, sort=False)[values_drop].sum()
            )
        ),
        row=1,
        col=2
    )
    fig.update_traces(
        hoverinfo="label+value+percent",
        textinfo="percent",
        textposition="inside"
    )
    fig.update_layout(
        #uniformtext_minsize=12,
        #uniformtext_mode="hide",
        height=500,
        width=1400,
        template="simple_white",
        showlegend=True
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True)
import numpy as np
import pandas as pd
#import plotly.express as px
import plotly.graph_objects as go
import requests
import shap
from dash import Dash, dcc, html, Input, Output, exceptions
from plotly.subplots import make_subplots

client = pd.read_csv("client_test.csv")

app = Dash(__name__)

shap.initjs()

app.layout = html.Div([
    html.Div([
        html.H1("Client credit score"),
        html.H4("Enter loan or client ID here:"),
        dcc.Input(id="id_input", type="number", debounce=True,
                  min=100001, max=999999),
        html.H4("Model response:"),
        html.H4(id="id_output"),
        html.H4(id="id_error", style={"color": "red"})
    ]),
    html.Div([
        html.H4("Features importance with SHAP:"),
        html.Div(id="json_output")
    ]),
    html.Div([
        html.H1("Client EDA"),
        """Our data scientist is working on it ;-)"""
    ]),
    html.Div([
        html.H1("Global EDA"),
        dcc.Graph(id="graph_01"),
        html.H4("Names"),
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
        html.H4("Values"),
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
])

@app.callback(
    Output("id_output", "children"),
    Output("id_error", "children"),
    Input("id_input", "value")
)
def call_score(id_input):
    if id_input is None:
        raise exceptions.PreventUpdate
    if id_input not in list(client["SK_ID_CURR"]):
        return "", "Not an existing ID !"
    client_ID = {"SK_ID_CURR": id_input, "threshold": 0.1}
    score = requests.post("http://127.0.0.1:8000/predict", json=client_ID)
    return "Prediction : {} | Probability : {}".format(
        score.json()["SCORE"], score.json()["PROBA"]
    ), ""

@app.callback(
    Output("json_output", "children"),
    Input("id_input", "value")
)
def call_features(id_input):
    if id_input is None:
        raise exceptions.PreventUpdate
    if id_input not in list(client["SK_ID_CURR"]):
        raise exceptions.PreventUpdate
    client_ID2 = {"SK_ID_CURR": id_input}
    features = requests.post("http://127.0.0.1:8000/features", json=client_ID2)
    plot = shap.force_plot(
        features.json()["explain_value"],
        np.array(features.json()["shap_values"]),
        np.array(features.json()["app_values"]),
        np.array(features.json()["features"])
    )
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    return html.Iframe(srcDoc=shap_html,
                       style={"width": "100%", "height": "200px", "border": 0})

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
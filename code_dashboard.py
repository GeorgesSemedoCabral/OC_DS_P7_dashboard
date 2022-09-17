import joblib
import numpy as np
import pandas as pd
#import plotly.express as px
import plotly.graph_objects as go
import requests
import shap
from dash import Dash, dcc, html, Input, Output, exceptions
from dash.dash_table import DataTable
from plotly.subplots import make_subplots

client = joblib.load("data/client_test.sav")

app = Dash(__name__)

server = app.server

blue_score = {"color": "blue"}
red_score = {"color": "red"}
purple_error = {"color": "purple"}

client_data = [
    "CODE_GENDER",
    "DAYS_BIRTH",
    "NAME_FAMILY_STATUS",
    "CNT_CHILDREN",
    "NAME_HOUSING_TYPE",
    "NAME_EDUCATION_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT"
]
client_rows = [
    "Gender",
    "Age",
    "Status",
    "Children",
    "Housing",
    "Education",
    "Occupation",
    "Organization",
    "Monthly income (€, 05-18-2018)",
    "Credit (€)"
]

app.layout = html.Div([
    html.Div([
        html.H3("Enter loan or client ID here:"),
        dcc.Input(id="id_input", type="number", debounce=True,
                  min=100001, max=999999),
        html.Button("Submit", id="input_button", n_clicks=0),
        html.H1("Client credit score"),
        html.H3("Model response:"),
        html.H4(id="id_output", style=blue_score),
    ]),
    html.Div([
        html.H3("Features importance with SHAP:"),
        html.Div(id="json_output")
    ]),
    html.Div([
        html.H1("Client information"),
        html.Div(id="table_output")
    ]),
        html.Div([
        html.H1("Client comparison"),
        """Our data scientist is working on it ;)"""
    ]),
    html.Div([
        html.H1("Global EDA"),
        dcc.Graph(id="graph_01"),
        html.H3("Names"),
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
        html.H3("Values"),
        dcc.Dropdown(
            id="values_drop",
            options=[
                "AMT_INCOME_TOTAL",
                "AMT_CREDIT",
            ],
            value="AMT_INCOME_TOTAL",
            clearable=False
        ),
    ])
])

@app.callback(
    Output("id_output", "children"),
    Output("id_output", "style"),
    Input("id_input", "value"),
    Input("input_button", "n_clicks")
)
def call_score(id_input, n_clicks):
    if n_clicks == 0:
        raise exceptions.PreventUpdate
    else:
        if id_input is None:
            raise exceptions.PreventUpdate
        if id_input not in list(client["SK_ID_CURR"]):
            return "Not an existing ID !", purple_error
        client_ID = {"SK_ID_CURR": id_input, "threshold": 0.1}
        score = requests.post(
            "https://oc-ds-p7-api.herokuapp.com/predict",
            json=client_ID
        )
        if score.json()["SCORE"] == "G":
            return "Prediction : {} | Probability : {}".format(
                score.json()["SCORE"], score.json()["PROBA"]
            ), blue_score
        elif score.json()["SCORE"] == "B":
            return "Prediction : {} | Probability : {}".format(
                score.json()["SCORE"], score.json()["PROBA"]
            ), red_score

@app.callback(
    Output("json_output", "children"),
    Input("id_input", "value"),
    Input("input_button", "n_clicks")
)
def call_features(id_input, n_clicks):
    if n_clicks == 0:
        raise exceptions.PreventUpdate
    else:
        if id_input is None:
            raise exceptions.PreventUpdate
        if id_input not in list(client["SK_ID_CURR"]):
            raise exceptions.PreventUpdate
        client_ID2 = {"SK_ID_CURR": id_input}
        features = requests.post(
            "https://oc-ds-p7-api.herokuapp.com/features",
            json=client_ID2
        )
        plot = shap.force_plot(
            features.json()["explain_value"],
            np.array(features.json()["shap_values"]),
            np.array(features.json()["app_values"]),
            np.array(features.json()["features"])
        )
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        return html.Iframe(
            srcDoc=shap_html,
            style={"width": "200%", "height": "110px", "border": 0}
        )

@app.callback(
    Output("table_output", "children"),
    Input("id_input", "value"),
    Input("input_button", "n_clicks")
)
def generate_table(id_input, n_clicks):
    if n_clicks == 0:
        raise exceptions.PreventUpdate
    else:
        if id_input is None:
            raise exceptions.PreventUpdate
        if id_input not in list(client["SK_ID_CURR"]):
            raise exceptions.PreventUpdate
        client_ID2 = {"SK_ID_CURR": id_input}
        id_client = client[client["SK_ID_CURR"]==client_ID2["SK_ID_CURR"]]
        df = pd.DataFrame({
            "INFO" : client_rows,
            "CLIENT" : list(
                id_client.loc[:, i].to_numpy().item(0) for i in client_data
            ),
        })
        return DataTable(
            data=df.to_dict("records"),
            style_cell={"textAlign": "center"},
            fill_width=False
        )

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
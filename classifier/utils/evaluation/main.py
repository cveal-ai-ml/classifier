"""
Author: MINDFUL
Purpose: Create a dashboard for monitoring Deep Learning (DL) training
"""


import dash
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from dash import html, dcc, callback, Input, Output

from utils.data import load_loss, median_filter


@callback(Output("live-loss", "figure"),
          [Input("interval-component", "n_intervals"),
           Input(component_id="dd_loss", component_property="value")])
def show_loss(n, dd_name):
    """
    Visualize training and validation loss all models

    Parameters:
    - results (pd.DataFrame[str, float]): combined training or validation file

    Returns:
    - (plotly.express.line): Updated figure
    """

    x_tag = "iteration"
    y_tag = "total"

    # Gather: Training Results (Training, Validation)

    train = load_loss(path_root, "train")

    if dd_name == "iterations (median filter)":
        train = median_filter(train)
    elif dd_name == "epochs":
        train = median_filter(train, t_tag="epoch")
        x_tag = "epoch"

    fig_train = go.Figure()
    for current_key in train:
        x_vals = train[current_key][x_tag]
        y_vals = train[current_key][y_tag]
        fig_train.add_trace(go.Scatter(x=x_vals, y=y_vals,
                                       mode="lines", name=current_key))

        fig_train.update_layout(xaxis_title="Iterations",
                                yaxis_title="Measure")

    try:

        valid = load_loss(path_root, "valid")

        fig_valid = go.Figure()
        for current_key in valid:
            x_vals = train[current_key][x_tag]
            y_vals = train[current_key][y_tag]
            fig_valid.add_trace(go.Scatter(x=x_vals, y=y_vals,
                                mode="lines", name=current_key))

            fig_valid.update_layout(xaxis_title="Iterations",
                                    yaxis_title="Measure")

        figures = [fig_train, fig_valid]

        all_titles = ["Train Loss", "Valid Loss"]

        fig = make_subplots(rows=len(figures), cols=1, subplot_titles=all_titles)

        for i, figure in enumerate(figures):
            for trace in range(len(figure["data"])):
                if i != len(figures) - 1:
                    figure["data"][trace].showlegend = False
                fig.append_trace(figure["data"][trace], row=i+1, col=1)

            fig["layout"]["xaxis%s" % (i+1)]["title"] = "Iterations"
            fig["layout"]["yaxis%s" % (i+1)]["title"] = "Error"

        fig.update_layout(autosize=False, height=1000)

    except Exception:

        fig = fig_train

    return fig


if __name__ == "__main__":

    path_root = "/Users/slane/Documents/research/results/classifier"

    # Create: Dashboard Layout

    dd_loss = ["iterations", "iterations (median filter)", "epochs"]

    style = {"textAlign": "center", "marginTop": 40, "marginBottom": 40}

    app = dash.Dash()

    app.layout = html.Div(id="parent",
                          children=[
                              html.H1(id="train", style=style,
                                      children="Train Loss"),
                              dcc.Dropdown(dd_loss, dd_loss[0], id="dd_loss"),
                              dcc.Graph(id="live-loss"),
                              dcc.Interval(id="interval-component",
                                           interval=5*1000,
                                           n_intervals=0)
                              ]
                          )

    # Start: Dashboard Server

    app.run_server(debug=True)

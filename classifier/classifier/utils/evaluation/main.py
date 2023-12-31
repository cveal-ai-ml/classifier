"""
Author: MINDFUL
Purpose: Create a dashboard for monitoring Deep Learning (DL) training
"""


import dash
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from dash import html, dcc, callback, Input, Output

from utils.data import load_loss, format_data


@callback(Output("live-loss", "figure"),
          [Input("interval-component", "n_intervals"),
           Input(component_id="dd_loss", component_property="value")])
def show_loss(n, dd_name, font_size=20):
    """
    Visualize training and validation loss all models

    Parameters:
    - results (pd.DataFrame[str, float]): combined training or validation file

    Returns:
    - (plotly.express.line): Updated figure
    """

    fig_height = 3000

    x_tag = "step"
    y_title = "Error"
    x_title = "Iterations"
    all_titles = ["Train Loss", "Valid Loss",
                  "Valid Accuracy", "LR - AdamW"]

    all_y_tags = ["train_error_step", "valid_error_step",
                  "accuracy_step", "lr-AdamW"]

    font = dict(family="Courier New, monospace",
                size=font_size, color="RebeccaPurple")

    # Gather: Training Results (Training, Validation)

    data = load_loss(path_root)

    use_filter = 0
    if dd_name == "iterations (median filter)":
        use_filter = 1

    elif dd_name == "epoch":
        use_filter = 2
        x_tag = "epoch"
        x_title = "Epochs"
        all_y_tags = [ele.replace("step", x_tag) for ele in all_y_tags]

    # Gather: Individual Plots

    figures = []
    max_values = []
    for i, y_tag in enumerate(all_y_tags):

        fig = go.Figure()
        local_values = []
        for current_key in data:

            if "lr" in y_tag:
                x_tag = "epoch"

            y_vals, x_vals = format_data(data[current_key],
                                         y_tag, x_tag, use_filter)

            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines",
                                     name=all_titles[i] + " " + current_key))

            local_values.append(max(y_vals))

        if len(local_values) == 0:
            value = 1
        else:
            value = max(local_values)

        max_values.append(value)

        figures.append(fig)

    # Plot: Results (Subplot)

    indices = [1 if len(ele["data"]) != 0 else 0 for ele in figures]

    if sum(indices) == 0:
        fig = go.Figure()
    else:
        if sum(indices) == 1:
            height = fig_height // 1.5
            index = indices.index(1)
            fig = figures[index]
            fig.update_layout(title_text=all_titles[index], title_x=0.5,
                              xaxis_title=x_title, yaxis_title=y_title)
        else:
            height = fig_height
            fig = make_subplots(rows=len(figures), cols=1,
                                x_title=x_title, y_title=y_title,
                                subplot_titles=all_titles,
                                vertical_spacing=0.04)

            for i, figure in enumerate(figures):
                for trace in range(len(figure["data"])):

                    fig.append_trace(figure["data"][trace], row=i+1, col=1)

        fig.update_annotations(font=font)
        fig.update_layout(font=font, height=height)

        for i, value in enumerate(max_values):
            if "lr" in all_y_tags[i]:
                continue
            y_lim = [-0.05, value + value * 0.1]
            fig.update_yaxes(range=y_lim, row=i+1, col=1)

    return fig


if __name__ == "__main__":

    path_root = "/develop/results/classifier/many_exps/cifar"

    # Create: Dashboard Layout

    dd_loss = ["iterations", "iterations (median filter)", "epoch"]

    style = {"textAlign": "center", "marginTop": 40, "marginBottom": 40}

    app = dash.Dash()

    app.layout = html.Div(id="parent",
                          children=[
                              html.H1(id="train", style=style,
                                      children="Classifier Analytics"),
                              dcc.Dropdown(dd_loss, dd_loss[0], id="dd_loss"),
                              dcc.Graph(id="live-loss"),
                              dcc.Interval(id="interval-component",
                                           interval=1*60000,
                                           n_intervals=0)
                              ]
                          )

    # Start: Dashboard Server

    app.run_server(debug=True)

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

    fig_height = 800

    x_tag = "step"
    y_title = "Error"
    x_title = "Iterations"
    all_titles = ["Train Loss", "Valid Loss", "Valid Accuracy"]
    all_y_tags = ["train_error_step", "valid_error_step", "accuracy_step"]

    font = dict(family="Courier New, monospace",
                size=font_size, color="RebeccaPurple")

    # Gather: Training Results (Training, Validation)

    data = load_loss(path_root)

    use_filter = 0
    if dd_name == "iterations (median filter)":
        use_filter = 1

    # Gather: Individual Plots

    figures = []
    max_values = []
    for y_tag in all_y_tags:

        fig = go.Figure()
        for current_key in data:

            try:
                data[current_key][y_tag]
            except Exception:
                continue

            y_vals, x_vals = format_data(data[current_key],
                                         y_tag, x_tag, use_filter)

            fig.add_trace(go.Scatter(x=x_vals, y=y_vals,
                                     mode="lines", name=current_key))

            max_values.append(max(y_vals))

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
                                vertical_spacing=0.1)

            for i, figure in enumerate(figures):
                for trace in range(len(figure["data"])):
                    if i != len(figures) - 1:
                        figure["data"][trace].showlegend = False
                    fig.append_trace(figure["data"][trace], row=i+1, col=1)

                    # fig["layout"]["xaxis%s" % (i+1)]["title"] = "Iterations"
                    # fig["layout"]["yaxis%s" % (i+1)]["title"] = "Error"

        fig.update_annotations(font=font)
        fig.update_layout(font=font, height=height)

        for i, value in enumerate(max_values):
            y_lim = [-0.05, value + value * 0.1]
            fig.update_yaxes(range=y_lim, row=i+1, col=1)

    return fig


if __name__ == "__main__":

    path_root = "/Users/slane/Documents/research/results/classifier"

    # Create: Dashboard Layout

    dd_loss = ["iterations", "iterations (median filter)"]

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

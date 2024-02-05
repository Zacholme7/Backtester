import os
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_orders():
    """
    Read in all of the generated orders
    """
    with open("./src/data/orders.json") as file:
        orders = json.load(file)
        return orders


def read_data():
    """
    Read in the candles that we backtested on
    """
    path = os.path.join("src", "data", "data.csv")
    return pd.read_csv(path)


def plot_data(df, strat_config, orders):
    """
    Plot the candles and the orders
    """
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02,
                        subplot_titles=("Candles"),
                        row_heights=[1.0])
    config = {'scrollZoom': True}

    # add the candles
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)

    for order in orders:
        date = order["date"]
        price = order["price"]
        trade_type = order["trade_type"]
        direction = order["direction"]
        if direction == "LONG":
            color = "green"
        else:
            color = "red"

        if trade_type == "OPEN":
            fig.add_trace(go.Scatter(
                x=[date],
                y=[price],
                mode="markers",
                marker=dict(color=color, size=13)
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=[date],
                y=[price],
                mode="markers",
                marker=dict(color=color, size=10, symbol='x')
            ), row=1, col=1)

    # update display
    fig.update_xaxes(
        showspikes=True,
        spikesnap="cursor",
        spikemode="across",
        spikethickness=1,
        spikecolor="grey",
        row=1, col=1
    )

    # general layout updates
    fig.update_layout(
        title=strat_config,
        xaxis_rangeslider_visible=False,
        height=1000,
        hovermode='x',
    )
    fig.show(config=config)


if __name__ == "__main__":
    # read in the orders
    orders = parse_orders()

    # read in the candle data
    data = read_data()

    # plot the data
    # plot_data(data, None, orders)

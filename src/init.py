import os
import json
import yaml
import numpy as np
import pandas as pd
import datetime
from itertools import product
from oandapyV20 import API
from oandapyV20.contrib.factories import InstrumentsCandlesFactory


def fetch_data(start_date, end_date, instr, granularity):
    """
    This function will fetch all the data from the the provided start date up until the last hour
    """
    data_start_date = datetime.datetime(2021, 1, 3).strftime("%Y-%m-%dT%H:%M:%SZ")

    csv_path = os.path.join("src", "data", "data.csv")
    if not os.path.exists(csv_path):
        # the data does not exist
        client = API(access_token=os.getenv("ACCESS_TOKEN"))

        # construct the request param
        params = {
            "granularity": granularity,
            "from": data_start_date,
            "to": end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        }

        candle_data = []

        # make all the request and process the data
        for r in InstrumentsCandlesFactory(instrument=instr, params=params):
            rv = client.request(r)

            for candle in rv.get("candles"):
                ctime = candle.get("time")[0:19]
                if candle["complete"]:
                    rec = "{time},{o},{h},{l},{c}".format(
                        time=ctime,
                        o=candle['mid']['o'],
                        h=candle['mid']['h'],
                        l=candle['mid']['l'],
                        c=candle['mid']['c'],
                    )
                    candle_data.append(rec)

        # do all of our formatting
        df = pd.DataFrame(candle_data, columns=['Combined'])
        df = df['Combined'].str.split(',', expand=True)
        df.columns = ["date", "open", "high", "low", "close"]
        index = df.index[df["date"] == start_date.strftime("%Y-%m-%dT%H:%M:%S")][0] - 3000 - 1
        df = df[index:]
        df.reset_index(drop=True, inplace=True)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        # Shift the 'close' column and use it as the new 'open' column
        df['open'] = df['close'].shift(1)
        # Convert 'Date' to datetime format
        df['date'] = pd.to_datetime(df['date'])
        df.fillna(method="bfill", inplace=True)

        # finally, save it to the csv
        df.to_csv(csv_path, index=False)


def load_yaml():
    """
    This function will read in the configuration file and set environment variables for use
    """
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
        for key, value in config.items():
            os.environ[key] = str(value)


def generate_config():
    """
    Generate the configuration for the backtester
    """

    def env_int(key):
        """
        internal helper
        """
        return int(os.getenv(key))

    def get_param(feature_number, param_number):
        access = f"FEATURE{feature_number}_PARAM{param_number}_"
        return (env_int(access + "START"), env_int(access + "STOP") + 1, env_int(access + "STEP"))

    # Feature settings
    feature_settings = {
        "RSI": {'param1_range': get_param(1, 1), 'param2_range': get_param(1, 2)},
        "CCI": {'param1_range': get_param(2, 1), 'param2_range': get_param(2, 2)},
        "ADX": {'param1_range': get_param(3, 1), 'param2_range': get_param(3, 2)},
        "WT": {'param1_range': get_param(4, 1), 'param2_range': get_param(4, 2)}
    }

    # The number of features we want to use
    number_of_features = env_int("NUM_FEATURES")

    # Select features based on the number of features specified
    selected_features = list(feature_settings.keys())[:number_of_features]

    # Generate parameter lists for each selected feature
    param_lists = []
    for feature in selected_features:
        param1_range = feature_settings[feature]['param1_range']
        param2_range = feature_settings[feature]['param2_range']
        param_list = [{'name': feature, 'param1': p1, 'param2': p2} for p1 in range(*param1_range) for p2 in range(*param2_range)]
        param_lists.append(param_list)

    # Regime settings
    regime_range = (float(os.getenv("REGIME_START")), float(os.getenv("REGIME_STOP")) + .1, float(os.getenv("REGIME_STEP")))
    regime_list = [round(p, 1) for p in np.arange(*regime_range)]

    # Kernel settings
    lookback_window = (env_int("LOOKBACK_WINDOW_START"), env_int("LOOKBACK_WINDOW_STOP") + 1, env_int("LOOKBACK_WINDOW_STEP"))
    weighting = (env_int("WEIGHT_START"), env_int("WEIGHT_STOP") + 1, env_int("WEIGHT_STEP"))
    regression_level = (env_int("REGRESSION_START"), env_int("REGRESSION_STOP") + 1, env_int("REGRESSION_STEP"))

    # Generate all combinations
    all_combinations = []
    for feature_params in product(*param_lists):
        for regime, lookback, weight, regression in product(regime_list, range(*lookback_window), range(*weighting), range(*regression_level)):
            feature_config = {"feature{}".format(i + 1): param for i, param in enumerate(feature_params)}
            config = {
                **feature_config,
                "regime": regime,
                "lookback": lookback,
                "weight": weight,
                "regression": regression,
                "num_features": number_of_features
            }
            all_combinations.append(config)

    # Write to a file
    with open("src/data/backtest_config.json", 'w') as file:
        json.dump(all_combinations, file)

    return all_combinations


if __name__ == "__main__":
    # load in all of the configurations
    load_yaml()

    # generate the configuration for the backtester
    generate_config()

    # retrieve the backtest data
    start_date = datetime.datetime.strptime(os.getenv("START_DATE"), "%Y-%m-%d")
    end_date = datetime.datetime.strptime(os.getenv("END_DATE"), "%Y-%m-%d")
    granularity = os.getenv("GRANULARITY")
    instrument = os.getenv("INSTR")
    fetch_data(start_date, end_date, instrument, granularity)


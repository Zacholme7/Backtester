use rayon::prelude::*;
use std::error::Error;
use std::fs;
mod backtester;
mod features;
mod strategy;
mod util;

extern crate ta_lib_wrapper;
use crate::backtester::backtest;
use crate::util::{read_csv, Config};

fn main() -> Result<(), Box<dyn Error>> {
    // data paths
    let candles_path = "../data/data.csv";
    let config_path = "../data/backtest_config.json";

    // read in the candles and the configs that we want to use
    let candles = read_csv(candles_path)?;
    let configs: Vec<Config> = serde_json::from_str(&fs::read_to_string(config_path)?)?;

    //process all of the configs in paralle
    configs.into_par_iter().for_each(|config| {
        backtest(&candles, config);
    });

    Ok(())
}

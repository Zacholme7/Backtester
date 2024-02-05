use crate::features::{
    cci, gaussian_kernel, generate_y_train, rational_quadratic_kernel, regime_filter, rsi,
    volatility_filter,
};
use polars::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct FeatureParam {
    pub name: String,
    pub param1: usize,
    pub param2: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
    pub num_features: usize,
    pub feature1: Option<FeatureParam>,
    pub feature2: Option<FeatureParam>,
    pub feature3: Option<FeatureParam>,
    pub feature4: Option<FeatureParam>,
    pub regime: f64,
    pub lookback: usize,
    pub weight: usize,
    pub regression: usize,
}

pub fn read_csv(path: &str) -> Result<DataFrame, Box<dyn std::error::Error>> {
    let candles = CsvReader::from_path(path)?
        .infer_schema(None)
        .has_header(true)
        .finish()?;
    Ok(candles)
}

pub fn calculate_feature(
    feature: &Option<FeatureParam>,
    candles: &DataFrame,
) -> Result<Series, Box<dyn std::error::Error>> {
    match feature {
        Some(feature_config) => {
            match feature_config.name.as_str() {
                "RSI" => rsi(candles, feature_config.param1, feature_config.param2),
                "CCI" => cci(candles, feature_config.param1, feature_config.param2),
                // Other feature calculations can be added here
                _ => Err("Unsupported feature".into()),
            }
        }
        None => Err("Feature not configured".into()),
    }
}

pub fn pad_to_max(
    features: &mut [Option<Series>],
    regime: &mut Series,
    volatility: &mut Series,
    y_train: &mut Series,
    max_len: usize,
) {
    for feature in features.iter_mut() {
        pad_series_option(feature, max_len);
    }

    pad_series(regime, max_len);
    pad_series(volatility, max_len);
    pad_series(y_train, max_len);
}

fn pad_series_option(series_option: &mut Option<Series>, max_length: usize) {
    if let Some(series) = series_option {
        pad_series(series, max_length);
    }
}

fn pad_series(series: &mut Series, max_length: usize) {
    let current_length = series.len();
    if current_length < max_length {
        let padding_length = max_length - current_length;

        // Check the data type and pad accordingly
        match series.dtype() {
            DataType::Int64 => {
                let padding_values = vec![0i64; padding_length];
                let mut new_data = Series::new(series.name(), &padding_values);
                new_data.append(series).unwrap();
                *series = new_data;
            },
            DataType::Float64 => {
                let padding_values = vec![f64::NAN; padding_length];
                let mut new_data = Series::new(series.name(), &padding_values);
                new_data.append(series).unwrap();
                *series = new_data;
            },
            DataType::Boolean => {
                let padding_values = vec![false; padding_length];
                let mut new_data = Series::new(series.name(), &padding_values);
                new_data.append(series).unwrap();
                *series = new_data;
            },
            _ => {
                // Handle or ignore other data types
                // You might want to log a warning or return an error
            }
        }
    }
}


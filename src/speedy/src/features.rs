use polars::prelude::*;
extern crate ta_lib_wrapper;
use ta_lib_wrapper::{TA_Integer, TA_Real, TA_RetCode, TA_ATR, TA_CCI, TA_EMA, TA_RSI};

pub fn rsi(
    candles: &DataFrame,
    n1: usize,
    n2: usize,
) -> Result<Series, Box<dyn std::error::Error>> {
    let close_prices = candles.column("close")?.f64()?;
    let close_prices_vec: Vec<f64> = close_prices.into_iter().filter_map(|x| x).collect();

    let mut rsi_out: Vec<TA_Real> = Vec::with_capacity(close_prices_vec.len());
    let mut rsi_out_begin: TA_Integer = 0;
    let mut rsi_out_size: TA_Integer = 0;

    unsafe {
        let ret_code = TA_RSI(
            0,
            close_prices_vec.len() as i32 - 1,
            close_prices_vec.as_ptr(),
            n1 as i32,
            &mut rsi_out_begin,
            &mut rsi_out_size,
            rsi_out.as_mut_ptr(),
        );
        match ret_code {
            TA_RetCode::TA_SUCCESS => rsi_out.set_len(rsi_out_size as usize),
            _ => return Err("failed to compute rsi".into()),
        }
    }

    let mut ema_rsi_out: Vec<TA_Real> = Vec::with_capacity(rsi_out_size as usize);
    let mut ema_out_begin: TA_Integer = 0;
    let mut ema_out_size: TA_Integer = 0;

    // Calculate EMA of RSI
    unsafe {
        let ret_code = TA_EMA(
            0,
            rsi_out.len() as i32 - 1,
            rsi_out.as_ptr(),
            n2 as i32,
            &mut ema_out_begin,
            &mut ema_out_size,
            ema_rsi_out.as_mut_ptr(),
        );
        match ret_code {
            TA_RetCode::TA_SUCCESS => ema_rsi_out.set_len(ema_out_size as usize),
            _ => return Err("failed to compute rsi".into()),
        }
    }

    let ema_rsi = Series::new("rsi", ema_rsi_out);
    Ok(rescale(&ema_rsi, 0.0, 100.0, 0.0, 1.0)?)
}

pub fn cci(
    candles: &DataFrame,
    n1: usize,
    n2: usize,
) -> Result<Series, Box<dyn std::error::Error>> {
    let close_prices = candles.column("close")?.f64()?;
    let close_prices_vec: Vec<f64> = close_prices.into_iter().filter_map(|x| x).collect();

    let mut cci_out: Vec<TA_Real> = Vec::with_capacity(close_prices_vec.len());
    let mut cci_out_begin: TA_Integer = 0;
    let mut cci_out_size: TA_Integer = 0;

    unsafe {
        let ret_code = TA_CCI(
            0,
            close_prices_vec.len() as i32 - 1,
            close_prices_vec.as_ptr(),
            close_prices_vec.as_ptr(),
            close_prices_vec.as_ptr(),
            n1 as i32,
            &mut cci_out_begin,
            &mut cci_out_size,
            cci_out.as_mut_ptr(),
        );
        match ret_code {
            TA_RetCode::TA_SUCCESS => cci_out.set_len(cci_out_size as usize),
            _ => panic!("could not compute rsi"),
        }
    }

    let mut ema_cci_out: Vec<TA_Real> = Vec::with_capacity(cci_out_size as usize);
    let mut ema_out_begin: TA_Integer = 0;
    let mut ema_out_size: TA_Integer = 0;

    // Calculate EMA of RSI
    unsafe {
        let ret_code = TA_EMA(
            0,
            cci_out.len() as i32 - 1,
            cci_out.as_ptr(),
            n2 as i32,
            &mut ema_out_begin,
            &mut ema_out_size,
            ema_cci_out.as_mut_ptr(),
        );
        match ret_code {
            TA_RetCode::TA_SUCCESS => ema_cci_out.set_len(ema_out_size as usize),
            _ => panic!("could not compute ema"),
        }
    }

    //Ok(Series::new("cci", ema_cci_out))
    Ok(Series::new("cci", normalize(&ema_cci_out, 0.0, 1.0)))
}
fn normalize(series: &[f64], min: f64, max: f64) -> Vec<f64> {
    let mut historic_min: f64 = 10e10;
    let mut historic_max: f64 = -10e10;

    for &value in series {
        if value.is_normal() {
            historic_min = historic_min.min(value);
            historic_max = historic_max.max(value);
        }
    }

    series
        .iter()
        .map(|&value| {
            if value.is_normal() {
                min + (max - min) * (value - historic_min)
                    / (historic_max - historic_min).max(10e-10)
            } else {
                value // Handle NaN or Infinity
            }
        })
        .collect()
}

pub fn regime_filter(df: &DataFrame, threshold: f64) -> Result<Series, Box<dyn std::error::Error>> {
    let opens = df.column("open")?.f64()?;
    let highs = df.column("high")?.f64()?;
    let lows = df.column("low")?.f64()?;
    let closes = df.column("close")?.f64()?;

    let src: Vec<f64> = opens
        .into_iter()
        .zip(highs)
        .zip(lows)
        .zip(closes)
        .map(|(((open, high), low), close)| {
            (open.unwrap_or(0.0) + high.unwrap_or(0.0) + low.unwrap_or(0.0) + close.unwrap_or(0.0))
                / 4.0
        })
        .collect();

    let length = src.len();
    let mut result = Series::new("regime_filter", vec![true; length]);

    let mut value1 = vec![0.0; length];
    let mut value2 = vec![0.0; length];
    let mut klmf = vec![0.0; length];

    for i in 1..length {
        value1[i] = 0.2 * (src[i] - src[i - 1]) + 0.8 * value1[i - 1];
        value2[i] =
            0.1 * (highs.get(i).unwrap_or(0.0) - lows.get(i).unwrap_or(0.0)) + 0.8 * value2[i - 1];
        let omega = if value2[i] != 0.0 {
            value1[i].abs() / value2[i].abs()
        } else {
            0.0
        };
        let alpha = (-omega.powi(2) + (omega.powi(4) + 16.0 * omega.powi(2)).sqrt()) / 8.0;
        klmf[i] = alpha * src[i] + (1.0 - alpha) * klmf[i - 1];
        //println!("{}, {}, {}", df.column("date")?.str()?.get(i).unwrap(), value1[i], value2[i]);
    }

    let abs_curve_slope: Vec<f64> = klmf
        .iter()
        .zip(klmf.iter().skip(1))
        .map(|(prev, curr)| (curr - prev).abs())
        .collect();

    let alpha_ema = 2.0 / (200.0 + 1.0);
    let mut exp_avg_abs_curve_slope = vec![0.0; length];
    exp_avg_abs_curve_slope[0] = abs_curve_slope[0];
    for i in 1..length - 1 {
        exp_avg_abs_curve_slope[i] =
            alpha_ema * abs_curve_slope[i] + (1.0 - alpha_ema) * exp_avg_abs_curve_slope[i - 1];
    }

    let normalized_slope_decline: Vec<f64> = abs_curve_slope
        .iter()
        .zip(exp_avg_abs_curve_slope.iter())
        .enumerate()
        .map(|(i, (acs, eascs))| {
            if i == 0 {
                0.0 // Handle the first element case
            } else {
                (abs_curve_slope[i - 1] - exp_avg_abs_curve_slope[i - 1])
                    / exp_avg_abs_curve_slope[i - 1]
            }
        })
        .collect();

    // Determine the boolean condition for each element
    let regime_filter_results: Vec<bool> = normalized_slope_decline
        .iter()
        .map(|&val| val >= threshold)
        .collect();

    result = Series::new("regime_filter", regime_filter_results);

    Ok(result)
}

pub fn volatility_filter(
    candles: &DataFrame,
    min_length: usize,
    max_length: usize,
) -> Result<Series, PolarsError> {
    let high: Vec<f64> = candles
        .column("high")?
        .f64()?
        .into_iter()
        .filter_map(|x| x)
        .collect();
    let low: Vec<f64> = candles
        .column("low")?
        .f64()?
        .into_iter()
        .filter_map(|x| x)
        .collect();
    let close: Vec<f64> = candles
        .column("close")?
        .f64()?
        .into_iter()
        .filter_map(|x| x)
        .collect();

    let mut recent_atr: Vec<TA_Real> = Vec::with_capacity(close.len());
    let mut recent_atr_begin: TA_Integer = 0;
    let mut recent_atr_size: TA_Integer = 0;

    unsafe {
        let ret_code = TA_ATR(
            0,
            close.len() as i32 - 1,
            high.as_ptr(),
            low.as_ptr(),
            close.as_ptr(),
            min_length as i32,
            &mut recent_atr_begin,
            &mut recent_atr_size,
            recent_atr.as_mut_ptr(),
        );
        match ret_code {
            TA_RetCode::TA_SUCCESS => recent_atr.set_len(recent_atr_size as usize),
            _ => panic!("unable to calculate recent ATR"),
        }
    }

    let mut historical_atr: Vec<TA_Real> = Vec::with_capacity(close.len());
    let mut historical_atr_begin: TA_Integer = 0;
    let mut historical_atr_size: TA_Integer = 0;

    unsafe {
        let ret_code = TA_ATR(
            0,
            close.len() as i32 - 1,
            high.as_ptr(),
            low.as_ptr(),
            close.as_ptr(),
            max_length as i32,
            &mut historical_atr_begin,
            &mut historical_atr_size,
            historical_atr.as_mut_ptr(),
        );
        match ret_code {
            TA_RetCode::TA_SUCCESS => historical_atr.set_len(historical_atr_size as usize),
            _ => panic!("unable to calculate historical ATR"),
        }
    }

    let mut recent_atr_series = Series::new("volatility", recent_atr);
    let historical_atr_series = Series::new("historical_atr", historical_atr);

    // Check the length difference and slice the longer series
    let length_diff = recent_atr_series.len() - historical_atr_series.len();
    if length_diff > 0 {
        // Slice recent_atr_series to match the size of historical_atr_series
        recent_atr_series =
            recent_atr_series.slice(length_diff as i64, historical_atr_series.len());
    }

    Ok(recent_atr_series.gt(&historical_atr_series)?.into_series())
}

pub fn rational_quadratic_kernel(
    df: &DataFrame,
    lookback: f64,
    relative_weight: f64,
    start_at_bar: usize,
    index: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let close = df.column("close")?.f64()?;
    let y_values = close
        .slice(index as i64 - start_at_bar as i64 - 1, start_at_bar + 1)
        .reverse();

    let weights: Vec<f64> = (0..=start_at_bar)
        .map(|i| {
            (1.0 + (i as f64).powi(2) / (lookback.powi(2) * 2.0 * relative_weight))
                .powf(-relative_weight)
        })
        .collect();

    let current_weight: f64 = y_values
        .into_iter()
        .zip(weights.iter())
        .map(|(y, w)| y.unwrap_or(0.0) * w)
        .sum();

    let cumulative_weight: f64 = weights.iter().sum();

    Ok(if cumulative_weight != 0.0 {
        current_weight / cumulative_weight
    } else {
        0.0
    })
}

pub fn gaussian_kernel(
    df: &DataFrame,
    lookback: f64,
    start_at_bar: usize,
    index: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let close = df.column("close")?.f64()?;
    let y_values = close
        .slice(index as i64 - start_at_bar as i64 - 1, start_at_bar + 1)
        .reverse();

    let weights: Vec<f64> = (0..=start_at_bar)
        .map(|i| (-((i as f64).powi(2)) / (2.0 * lookback.powi(2))).exp())
        .collect();

    let current_weight: f64 = y_values
        .into_iter()
        .zip(weights.iter())
        .map(|(y, w)| y.unwrap_or(0.0) * w)
        .sum();

    let cumulative_weight: f64 = weights.iter().sum();

    Ok(if cumulative_weight != 0.0 {
        current_weight / cumulative_weight
    } else {
        0.0
    })
}

pub fn generate_y_train(df: &DataFrame) -> Result<Series, Box<dyn std::error::Error>> {
    let close_values: Vec<f64> = df
        .column("close")?
        .f64()?
        .into_iter()
        .map(|val| val.unwrap_or(0.0))
        .collect();

    let mut directions: Vec<i64> = vec![0; close_values.len()];

    for i in 4..close_values.len() {
        let difference = close_values[i] - close_values[i - 4];
        directions[i] = if difference > 0.0 {
            -1
        } else if difference < 0.0 {
            1
        } else {
            0
        }
    }

    Ok(Series::new("y_train", directions))
}

pub fn rescale(
    series: &Series,
    old_min: f64,
    old_max: f64,
    new_min: f64,
    new_max: f64,
) -> Result<Series, Box<dyn std::error::Error>> {
    let scaled_series = series
        .f64()?
        .into_iter()
        .map(|opt_value| {
            opt_value.map(|value| {
                let scale = (old_max - old_min).max(10e-10);
                new_min + (new_max - new_min) * (value - old_min) / scale
            })
        })
        .collect::<Vec<_>>();

    Ok(Series::new(series.name(), scaled_series))
}

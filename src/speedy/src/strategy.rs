use crate::features::{
    cci, gaussian_kernel, generate_y_train, rational_quadratic_kernel, regime_filter, rsi,
    volatility_filter,
};
use serde::{Serialize, Deserialize};
use crate::util::{calculate_feature, pad_to_max, Config, FeatureParam};
use polars::prelude::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Direction {
    LONG = 1,
    SHORT = -1,
    NEUTRAL = 0,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TradeType {
    OPEN,
    CLOSE,
}

#[derive(Debug)]
pub struct StrategyState {
    neighbor_count: usize,
    bars_back: usize,
    prev_signal: Direction,
    curr_signal: Direction,
    predictions: Vec<i64>,
    prediction: i64,
    is_long_active: bool,
    is_short_active: bool,
    yhat1: Vec<f64>,
    yhat2: Vec<f32>,
    bullish_change: Vec<bool>,
    bearish_change: Vec<bool>,
    valid_short_exit: Vec<bool>,
    valid_long_exit: Vec<bool>,
    start_long_trade: Vec<bool>,
    start_short_trade: Vec<bool>,
    last_price: f64,
    index: usize,
    distances: Vec<f64>,
}

impl StrategyState {
    fn new() -> Self {
        StrategyState {
            neighbor_count: 8,
            bars_back: 2000,
            prev_signal: Direction::NEUTRAL,
            curr_signal: Direction::NEUTRAL,
            predictions: Vec::new(),
            prediction: 0,
            is_long_active: false,
            is_short_active: false,
            yhat1: Vec::new(),
            yhat2: Vec::new(),
            bullish_change: Vec::new(),
            bearish_change: Vec::new(),
            valid_short_exit: Vec::new(),
            valid_long_exit: Vec::new(),
            start_long_trade: Vec::new(),
            start_short_trade: Vec::new(),
            last_price: 0.0,
            index: 0,
            distances: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct Strategy<'a> {
    config: Config,
    strategy_state: StrategyState,
    pub feature1_data: Option<ChunkedArray<Float64Type>>,
    pub feature2_data: Option<ChunkedArray<Float64Type>>,
    pub feature3_data: Option<ChunkedArray<Float64Type>>,
    pub feature4_data: Option<ChunkedArray<Float64Type>>,
    pub y_train: ChunkedArray<Int64Type>,
    pub regime: ChunkedArray<BooleanType>,
    pub volatility: ChunkedArray<BooleanType>,
    pub candles: &'a DataFrame,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub date: String,
    pub price: f64,
    pub trade_type: TradeType,
    pub direction: Direction,
}

impl<'a> Strategy<'a> {
    pub fn new(config: Config, candles: &'a DataFrame) -> Result<Self, Box<dyn std::error::Error>> {
        let feature_params = [
            &config.feature1,
            &config.feature2,
            &config.feature3,
            &config.feature4,
        ];
        let mut features = Vec::new();

        for (i, feature_param) in feature_params.iter().enumerate() {
            if i + 1 <= config.num_features as usize {
                features.push(Some(calculate_feature(feature_param, candles)?));
            } else {
                features.push(None);
            }
        }

        let mut y_train = generate_y_train(&candles)?;
        let mut regime = regime_filter(&candles, config.regime)?;
        let mut volatility = volatility_filter(&candles, 1, 10)?;

        pad_to_max(
            &mut features,
            &mut regime,
            &mut volatility,
            &mut y_train,
            candles.height(),
        );

        let strategy_state = StrategyState::new();

        Ok(Self {
            config,
            strategy_state,
            feature1_data: features[0].as_ref().map(|s| s.f64().unwrap()).cloned(),
            feature2_data: features[1].as_ref().map(|s| s.f64().unwrap()).cloned(),
            feature3_data: features[2].as_ref().map(|s| s.f64().unwrap()).cloned(),
            feature4_data: features[3].as_ref().map(|s| s.f64().unwrap()).cloned(),
            y_train: y_train.i64()?.clone(),
            regime: regime.bool()?.clone(),
            volatility: volatility.bool()?.clone(),
            candles,
        })
    }

    pub fn update_state(&mut self, index: usize) {
        let start_index = if index >= self.strategy_state.bars_back {
            index - self.strategy_state.bars_back
        } else {
            0
        };

        let size_loop = std::cmp::min(self.strategy_state.bars_back - 1, self.y_train.len() - 1);
        let mut last_distance = -1.0;

        if start_index > 0 {
            for i in (0..size_loop).step_by(4) {
                let d = self.get_lorentzian_distance(i, index);
                if d >= last_distance {
                    self.strategy_state.distances.push(d);
                    self.strategy_state
                        .predictions
                        .push(self.y_train.get(i).unwrap());
                    if self.strategy_state.predictions.len() > self.strategy_state.neighbor_count {
                        let x = (self.strategy_state.neighbor_count as f64 * 3.0 / 4.0).round()
                            as usize;
                        last_distance = self.strategy_state.distances[x];
                        self.strategy_state.distances.remove(0);
                        self.strategy_state.predictions.remove(0);
                    }
                }
            }

            self.strategy_state.prediction = self.strategy_state.predictions.iter().sum::<i64>();
        }

        let filter_all = self.volatility.get(index).unwrap() && self.regime.get(index).unwrap();
        self.strategy_state.prev_signal = self.strategy_state.curr_signal.clone();

        if self.strategy_state.prediction > 0 && filter_all {
            self.strategy_state.curr_signal = Direction::LONG;
        } else if self.strategy_state.prediction < 0 && filter_all {
            self.strategy_state.curr_signal = Direction::SHORT;
        }
    }

    pub fn get_lorentzian_distance(&self, i: usize, index: usize) -> f64 {
        let mut distance = 0.0;

        if let Some(feature1_data) = &self.feature1_data {
            distance += self.calculate_lorentzian_distance_for_feature(feature1_data, i, index);
        }

        if let Some(feature2_data) = &self.feature2_data {
            distance += self.calculate_lorentzian_distance_for_feature(feature2_data, i, index);
        }

        if let Some(feature3_data) = &self.feature3_data {
            distance += self.calculate_lorentzian_distance_for_feature(feature3_data, i, index);
        }

        if let Some(feature4_data) = &self.feature4_data {
            distance += self.calculate_lorentzian_distance_for_feature(feature4_data, i, index);
        }
        distance
    }

    fn calculate_lorentzian_distance_for_feature(
        &self,
        feature_data: &ChunkedArray<Float64Type>,
        i: usize,
        index: usize,
    ) -> f64 {
        feature_data
            .get(i)
            .zip(feature_data.get(index))
            .map(|(a, b)| (1.0 + (a - b).abs()).ln())
            .unwrap_or(0.0)
    }

    pub fn converge(&mut self, index: usize, candles: &DataFrame) -> Vec<Order> {
        self.update_state(index);
        let yhat1 = rational_quadratic_kernel(&candles, 8.0, 8.0, 25, index).unwrap();
        self.strategy_state.yhat1.push(yhat1);

        let mut to_execute: Vec<Order> = Vec::new();
        let (start_short, start_long, end_long, end_short) = self.generate_signals(index as i64);

        self.process_signals(
            &mut to_execute,
            index,
            start_long,
            start_short,
            end_long,
            end_short,
        );
        to_execute
    }

    fn process_signals(
        &mut self,
        to_execute: &mut Vec<Order>,
        index: usize,
        start_long: bool,
        start_short: bool,
        end_long: bool,
        end_short: bool,
    ) {
        if start_long {
            self.construct_start_long(to_execute, index);
        }
        if start_short {
            self.construct_start_short(to_execute, index);
        }
        if end_short {
            self.construct_end_short(to_execute, index);
        }
        if end_long {
            self.construct_end_long(to_execute, index);
        }
    }

    fn bars_since(&self, data: &Vec<bool>) -> usize {
        data.iter().rev().position(|&x| x).unwrap_or(0)
    }

    // Main method to generate signals
    fn generate_signals(&mut self, index: i64) -> (bool, bool, bool, bool) {
        let is_buy_signal = self.strategy_state.curr_signal == Direction::LONG;
        let is_sell_signal = self.strategy_state.curr_signal == Direction::SHORT;
        let is_new_buy_signal =
            is_buy_signal && self.strategy_state.curr_signal != self.strategy_state.prev_signal;
        let is_new_sell_signal =
            is_sell_signal && self.strategy_state.curr_signal != self.strategy_state.prev_signal;

        let scaled_index = if index >= self.strategy_state.bars_back as i64 {
            (index - self.strategy_state.bars_back as i64) as usize
        } else {
            return (false, false, false, false);
        };

        if scaled_index >= 3 {
            let was_bearish = self.strategy_state.yhat1[scaled_index - 2]
                > self.strategy_state.yhat1[scaled_index - 1];
            let was_bullish = self.strategy_state.yhat1[scaled_index - 2]
                <= self.strategy_state.yhat1[scaled_index - 1];
            let is_bearish = self.strategy_state.yhat1[scaled_index - 1]
                >= self.strategy_state.yhat1[scaled_index];
            let is_bullish = self.strategy_state.yhat1[scaled_index - 1]
                < self.strategy_state.yhat1[scaled_index];
            let is_bearish_change = is_bearish && was_bullish;
            let is_bullish_change = is_bullish && was_bearish;

            let start_long_trade = is_new_buy_signal && is_bullish;
            let start_short_trade = is_new_sell_signal && is_bearish;

            self.strategy_state.start_long_trade.push(start_long_trade);
            self.strategy_state
                .start_short_trade
                .push(start_short_trade);
            self.strategy_state.bearish_change.push(is_bearish_change);
            self.strategy_state.bullish_change.push(is_bullish_change);

            let is_valid_long_exit = self.bars_since(&self.strategy_state.bearish_change)
                > self.bars_since(&self.strategy_state.start_long_trade);
            let is_valid_short_exit = self.bars_since(&self.strategy_state.bullish_change)
                > self.bars_since(&self.strategy_state.start_short_trade);

            let end_long_trade = is_bearish_change
                && self
                    .strategy_state
                    .valid_long_exit
                    .last()
                    .copied()
                    .unwrap_or(false);
            let end_short_trade = is_bullish_change
                && self
                    .strategy_state
                    .valid_short_exit
                    .last()
                    .copied()
                    .unwrap_or(false);

            self.strategy_state
                .valid_short_exit
                .push(is_valid_short_exit);
            self.strategy_state.valid_long_exit.push(is_valid_long_exit);

            if start_long_trade && self.strategy_state.is_short_active {
                self.strategy_state.is_short_active = false;
                self.strategy_state.is_long_active = true;
            } else if start_short_trade && self.strategy_state.is_long_active {
                self.strategy_state.is_long_active = false;
                self.strategy_state.is_short_active = true;
            }


            (
                start_long_trade,
                start_short_trade,
                end_long_trade,
                end_short_trade,
            )
        } else {
            (false, false, false, false)
        }
    }

    fn construct_start_long(&mut self, to_execute: &mut Vec<Order>, index: usize) {
        self.record_trade(to_execute, index, TradeType::OPEN, 0.00007, Direction::LONG);
    }

    fn construct_start_short(&mut self, to_execute: &mut Vec<Order>, index: usize) {
        self.record_trade(to_execute, index, TradeType::OPEN, -0.00007, Direction::SHORT);
    }

    fn construct_end_long(&mut self, to_execute: &mut Vec<Order>, index: usize) {
        self.record_trade(to_execute, index, TradeType::CLOSE, -0.00007, Direction::LONG);
    }

    fn construct_end_short(&mut self, to_execute: &mut Vec<Order>, index: usize) {
        self.record_trade(to_execute, index, TradeType::CLOSE, 0.00007, Direction::SHORT);
    }

    fn record_trade(
        &mut self,
        to_execute: &mut Vec<Order>,
        index: usize,
        trade_type: TradeType,
        offset: f64,
        direction: Direction,
    ) {
        let trade_price = self
            .candles
            .column("close")
            .unwrap()
            .f64()
            .unwrap()
            .get(index)
            .unwrap()
            + offset;
        if trade_type == TradeType::OPEN {
            self.strategy_state.last_price = trade_price;
        }
        let new_order = Order {
            date: String::from(
                self.candles
                    .column("date")
                    .unwrap()
                    .str()
                    .unwrap()
                    .get(index)
                    .unwrap(),
            ),
            price: trade_price,
            trade_type,
            direction,
        };
        to_execute.push(new_order);
    }

    
}

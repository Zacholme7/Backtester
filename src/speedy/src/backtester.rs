use crate::strategy::{Direction, Order, Strategy, TradeType};
use crate::util::Config;
use polars::prelude::*;
use std::fs::File;
use std::io::prelude::*;

pub fn backtest(candles: &DataFrame, config: Config) {
    // construct the strategy and process all the data
    let mut strategy = Strategy::new(config, candles).expect("unable to create the strategy");

    // process each candle
    let mut all_orders = Vec::new();
    for i in 2000..candles.height() {
        all_orders.extend(strategy.converge(i, &candles))
    }

    println!("{}", calculate_pnl_in_pips(&all_orders));

    // serialize the orders and save them to file
    let serialized_orders = serde_json::to_string(&all_orders).unwrap();
    let mut file = File::create("../data/orders.json").unwrap();
    file.write_all(serialized_orders.as_bytes()).unwrap();
}

pub fn calculate_pnl_in_pips(orders: &[Order]) -> f64 {
    let mut total_pnl = 0.0;
    let mut open_order: Option<&Order> = None;
    let pip_value = 0.0001; // Adjust based on the currency pair and lot size

    for order in orders {
        match order.trade_type {
            TradeType::OPEN => open_order = Some(order),
            TradeType::CLOSE => {
                if let Some(open) = open_order {
                    let pnl = match open.direction {
                        Direction::LONG => order.price - open.price,
                        Direction::SHORT => open.price - order.price,
                        _ => continue
                    };
                    total_pnl += pnl;
                }
                open_order = None;
            }
        }
    }

    total_pnl / pip_value
}

#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import statsmodels.api as sm

INPUT_CSV = "sp500_close_prices.csv"
PAIRS_CSV = "cointegrated_pairs.csv"
PLOTS_DIR = "plots"
NUM_PAIRS_TO_TEST = 10

# Strategy parameters
WINDOW = 60  # Rolling window for z-score calc
ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 0.5
COMMISSION = 0.001  # 0.1% commission per trade


def backtest(pair_prices: pd.DataFrame, ticker_a: str, ticker_b: str) -> dict:
    # Get hedge ratio and spread
    y = pair_prices[ticker_a]
    x = sm.add_constant(pair_prices[ticker_b])
    model = sm.OLS(y, x).fit()
    hedge_ratio = model.params[1]
    spread = pair_prices[ticker_a] - hedge_ratio * pair_prices[ticker_b]

    # Get z-score
    rolling_mean = spread.rolling(window=WINDOW).mean()
    rolling_std = spread.rolling(window=WINDOW).std()
    z_score = (spread - rolling_mean) / rolling_std

    # Generate trading signals
    # Long signal (buy spread) when z-score is low
    long_entry = z_score < -ENTRY_THRESHOLD
    long_exit = z_score >= -EXIT_THRESHOLD
    # Short signal (sell spread) when z-score is high
    short_entry = z_score > ENTRY_THRESHOLD
    short_exit = z_score <= EXIT_THRESHOLD

    # Combine signals into a single positions series
    # 1 long, -1 short, 0 flat
    positions = pd.Series(np.nan, index=z_score.index)
    positions[long_entry] = 1
    positions[short_entry] = -1
    positions[long_exit & (positions.shift(1) == 1)] = 0
    positions[short_exit & (positions.shift(1) == -1)] = 0
    # Forward-fill positions to hold through periods
    positions.ffill(inplace=True)
    positions.fillna(0, inplace=True)

    # Get returns
    daily_returns = pair_prices.pct_change()
    # Returns are based on positions taken
    # For long position (1), long ticker_a and short ticker_b
    # For short position (-1), short ticker_a and long ticker_b
    portfolio_returns = positions.shift(1) * (
        daily_returns[ticker_a] - hedge_ratio * daily_returns[ticker_b]
    )
    # Include transaction costs
    # Trade occurs when position changes from previous day
    trades = positions.diff().abs()
    transaction_costs = trades * COMMISSION
    # Subtract costs from portfolio returns
    net_portfolio_returns = portfolio_returns - transaction_costs

    # Get cumulative returns (equity curve)
    cumulative_returns = (1 + net_portfolio_returns).cumprod()

    if cumulative_returns.iloc[-1] <= 0:
        total_return = -1.0
        cagr = -1.0  # -100% CAGR for total loss
        max_drawdown = -1.0  # -100% max drawdown
        sharpe_ratio = 0  # Sharpe ratio not meaningful here
    else:
        total_return = cumulative_returns.iloc[-1] - 1
        num_years = len(cumulative_returns) / 252
        cagr = (cumulative_returns.iloc[-1]) ** (1 / num_years) - 1
        sharpe_ratio = (
            (net_portfolio_returns.mean() / net_portfolio_returns.std()) * np.sqrt(252)
            if net_portfolio_returns.std() != 0
            else 0
        )
        cumulative_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

    # Max drawdown
    cumulative_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    return {
        "pair": (ticker_a, ticker_b),
        "total_return": total_return,
        "cagr": cagr,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "trades": int(trades.sum()),
    }


def main():
    # Load pair data
    try:
        pairs_df = pd.read_csv(PAIRS_CSV)
        # pair column is string like "('BRO', 'COST')", so evaluate it
        pairs_df["pair"] = pairs_df["pair"].apply(eval)
    except FileNotFoundError:
        print(
            f"Error: The file '{PAIRS_CSV}' was not found. Run the cointegration script first."
        )
        return

    # Load price data
    try:
        close_prices = pd.read_csv(INPUT_CSV, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV}' was not found.")
        return

    # Loop through top pairs and backtest
    print(f"Testing top {NUM_PAIRS_TO_TEST} cointegrated pairs...")
    results = []
    top_pairs = pairs_df.head(NUM_PAIRS_TO_TEST)

    for index, row in top_pairs.iterrows():
        ticker_a, ticker_b = row["pair"]
        print(f"\n--- Backtesting Pair: ({ticker_a}, {ticker_b}) ---")

        # Check if both tickers are in price data
        if ticker_a in close_prices.columns and ticker_b in close_prices.columns:
            pair_prices = close_prices[[ticker_a, ticker_b]].dropna()
            if not pair_prices.empty:
                performance = backtest(pair_prices, ticker_a, ticker_b)
                results.append(performance)
                print(
                    f"CAGR: {performance['cagr']:.2%}, Sharpe: {performance['sharpe_ratio']:.2f}, Max Drawdown: {performance['max_drawdown']:.2%}"
                )
        else:
            print(
                f"Skipping pair ({ticker_a}, {ticker_b}) - one or both tickers not in price data."
            )

    # Create final report
    if results:
        results_df = pd.DataFrame(results)
        # Sort by sharpe ratio for ranking
        results_df.sort_values(by="sharpe_ratio", ascending=False, inplace=True)

        print("\n\n--- Systematic Backtest Summary ---")
        print(f"Top {len(results)} pairs ranked by Sharpe Ratio:")
        print(
            results_df.to_string(
                index=False,
                formatters={
                    "total_return": "{:.2%}".format,
                    "cagr": "{:.2%}".format,
                    "max_drawdown": "{:.2%}".format,
                    "sharpe_ratio": "{:.2f}".format,
                },
            )
        )
        print("-----------------------------------")


if __name__ == "__main__":
    main()

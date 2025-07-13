#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

INPUT_CSV = "sp500_close_prices.csv"
PLOTS_DIR = "plots"
TICKER_A = "AMAT"
TICKER_B = "NXPI"

# Strategy parameters
ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 0.5
COMMISSION = 0.001


def kalman_filter_hedge_ratio(prices_a, prices_b):
    # Kalman Filter estimates hedge ratio based on stock prices (hidden state on observation)
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=0,
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01,
    )

    # Use filter to estimate hedge ratio over time
    state_means, _ = kf.filter(prices_a.values / prices_b.values)

    # Return as pandas series with correct index
    return pd.Series(state_means.flatten(), index=prices_a.index)


def backtest_kalman_strategy(file_path: str, ticker_a: str, ticker_b: str):
    # Load data
    print(f"Loading data for Kalman Filter backtest on ({ticker_a}, {ticker_b})...")
    try:
        close_prices = pd.read_csv(file_path, index_col=0, parse_dates=True)
        pair_prices = close_prices[[ticker_a, ticker_b]].dropna()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Calculate dynamic hedge ratio using KF
    print("Calculating dynamic hedge ratio with Kalman Filter...")
    hedge_ratio = kalman_filter_hedge_ratio(
        pair_prices[ticker_a], pair_prices[ticker_b]
    )
    # Calculate dynamic spread
    spread = pair_prices[ticker_a] - hedge_ratio * pair_prices[ticker_b]
    # Calculate z-score of spread
    rolling_mean = spread.rolling(window=60).mean()
    rolling_std = spread.rolling(window=60).std()
    z_score = (spread - rolling_mean) / rolling_std

    # Generate trading signals
    print("Generating trading signals...")
    # Long signal (buy spread) when z-score is low
    long_entry = z_score < -ENTRY_THRESHOLD
    long_exit = z_score >= -EXIT_THRESHOLD
    # Short signal (sell spread) when z-score is high
    short_entry = z_score > ENTRY_THRESHOLD
    short_exit = z_score <= EXIT_THRESHOLD

    positions = pd.Series(np.nan, index=z_score.index)
    positions[long_entry] = 1
    positions[short_entry] = -1
    positions[long_exit & (positions.shift(1) == 1)] = 0
    positions[short_exit & (positions.shift(1) == -1)] = 0
    positions.ffill(inplace=True)
    positions.fillna(0, inplace=True)

    # Get strategy returns with transaction costs
    print("Calculating strategy returns...")
    daily_returns = pair_prices.pct_change()
    # Hedge ratio is now a time-series and not a constant
    portfolio_returns = positions.shift(1) * (
        daily_returns[ticker_a] - hedge_ratio.shift(1) * daily_returns[ticker_b]
    )
    trades = positions.diff().abs()
    transaction_costs = trades * COMMISSION
    net_portfolio_returns = portfolio_returns - transaction_costs

    # Get performance metrics
    print("Calculating performance metrics...")
    cumulative_returns = (1 + net_portfolio_returns).cumprod()

    if cumulative_returns.empty or cumulative_returns.iloc[-1] <= 0:
        total_return, cagr, max_drawdown, sharpe_ratio = -1.0, -1.0, -1.0, 0
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

    # Plot final equity curve
    plt.figure(figsize=(15, 7))
    cumulative_returns.plot()
    plt.title(
        f"Kalman Filter Strategy Performance: {ticker_a} vs {ticker_b}", fontsize=16
    )
    plt.ylabel("Cumulative Returns")
    plt.xlabel("Date")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_filename = os.path.join(
        PLOTS_DIR, f"kalman_performance_{ticker_a}_{ticker_b}.png"
    )
    plt.savefig(plot_filename)
    plt.close()

    # Print report
    print("\n--- Kalman Filter Backtest Results ---")
    print(f"Pair: ({ticker_a}, {ticker_b})")
    print(f"Total Trades: {int(trades.sum())}")
    print(f"Total Strategy Return: {total_return:.2%}")
    print(f"Compound Annual Growth Rate (CAGR): {cagr:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Plot saved to '{plot_filename}'")
    print("--------------------------------------")


if __name__ == "__main__":
    backtest_kalman_strategy(INPUT_CSV, TICKER_A, TICKER_B)

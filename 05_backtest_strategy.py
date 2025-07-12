#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import statsmodels.api as sm

INPUT_CSV = "sp500_close_prices.csv"
PLOTS_DIR = "plots"
TICKER_A = "BRO"
TICKER_B = "COST"

# Strategy parameters
WINDOW = 60  # Rolling window for z-score calc
ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 0.5


def backtest_strategy(file_path: str, ticker_a: str, ticker_b: str):
    # Load data and get spread
    print(f"Loading data and calculating spread for ({ticker_a}, {ticker_b})...")
    try:
        close_prices = pd.read_csv(file_path, index_col=0, parse_dates=True)
        pair_prices = close_prices[[ticker_a, ticker_b]].dropna()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

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
    print("Generating trading signals...")
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

    # Get strategy returns
    print("Calculating strategy returns...")
    # Get daily returns for each stock
    daily_returns = pair_prices.pct_change()

    # Returns are based on positions taken
    # For long position (1), long ticker_a and short ticker_b
    # For short position (-1), short ticker_a and long ticker_b
    portfolio_returns = positions.shift(1) * (
        daily_returns[ticker_a] - hedge_ratio * daily_returns[ticker_b]
    )

    # Get cumulative returns (equity curve)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Plot
    print("Plotting performance...")
    plt.figure(figsize=(15, 7))
    cumulative_returns.plot()
    plt.title(
        f"Pairs Trading Strategy Performance: {ticker_a} vs {ticker_b}", fontsize=16
    )
    plt.ylabel("Cumulative Returns")
    plt.xlabel("Date")

    # Save plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_filename = os.path.join(
        PLOTS_DIR, f"backtest_performance_{ticker_a}_{ticker_b}.png"
    )
    plt.savefig(plot_filename)
    plt.close()

    # Print final metrics
    total_return = cumulative_returns.iloc[-1] - 1
    print("\n--- Backtest Results ---")
    print(f"Pair: ({ticker_a}, {ticker_b})")
    print(f"Total Strategy Return: {total_return:.2%}")
    print(f"Plot saved to '{plot_filename}'")
    print("------------------------")


if __name__ == "__main__":
    backtest_strategy(INPUT_CSV, TICKER_A, TICKER_B)

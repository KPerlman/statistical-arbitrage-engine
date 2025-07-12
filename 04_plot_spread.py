#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import statsmodels.api as sm

INPUT_CSV = "sp500_close_prices.csv"
PLOTS_DIR = "plots"

# Pair with lowest p-value
TICKER_A = "BRO"
TICKER_B = "COST"

plt.style.use("seaborn-v0_8-whitegrid")


def analyze_pair_spread(file_path: str, ticker_a: str, ticker_b: str):
    # Load data
    print(f"Loading data from '{file_path}'...")
    try:
        close_prices = pd.read_csv(file_path, index_col=0, parse_dates=True)
        # Check both tickers are in dataset
        if not all(ticker in close_prices.columns for ticker in [ticker_a, ticker_b]):
            print(
                f"Error: One or both tickers ({ticker_a}, {ticker_b}) not found in the dataset."
            )
            return
        pair_prices = close_prices[[ticker_a, ticker_b]].dropna()
        print(f"Analyzing spread for the pair: ({ticker_a}, {ticker_b})")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Get spread and hedge ratio (how many shares of B to short for each share of A)
    # Run linear regression to find hedge ratio
    y = pair_prices[ticker_a]
    x = sm.add_constant(pair_prices[ticker_b])  # Add constant for regression intercept

    model = sm.OLS(y, x).fit()
    hedge_ratio = model.params[1]

    # Spread is difference between two price series adjusted by hedge ratio
    spread = pair_prices[ticker_a] - hedge_ratio * pair_prices[ticker_b]
    print(f"Calculated hedge ratio: {hedge_ratio:.4f}")

    # Get z-score of spread (standardized spread) based on rolling mean and std
    # Using 60-day rolling window
    rolling_mean = spread.rolling(window=60).mean()
    rolling_std = spread.rolling(window=60).std()

    z_score = (spread - rolling_mean) / rolling_std

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Panel 1 with normalized prices
    (pair_prices / pair_prices.iloc[0] * 100).plot(ax=ax1)
    ax1.set_title(f"Normalized Prices for {ticker_a} and {ticker_b}", fontsize=16)
    ax1.set_ylabel("Normalized Price")
    ax1.legend()

    # Panel 2 with z-score of spread
    z_score.plot(ax=ax2)
    # Add horizontal lines for trading signals
    ax2.axhline(2.0, color="red", linestyle="--")
    ax2.axhline(-2.0, color="green", linestyle="--")
    ax2.axhline(0.0, color="black", linestyle="-")
    ax2.set_title("Z-Score of the Price Spread (60-Day Rolling)", fontsize=16)
    ax2.set_ylabel("Z-Score")
    ax2.set_xlabel("Date")

    plt.tight_layout()

    # Save plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_filename = os.path.join(PLOTS_DIR, f"spread_zscore_{ticker_a}_{ticker_b}.png")
    plt.savefig(plot_filename)
    print(f"\nAnalysis complete. Plot saved to '{plot_filename}'")
    plt.close()


if __name__ == "__main__":
    analyze_pair_spread(INPUT_CSV, TICKER_A, TICKER_B)

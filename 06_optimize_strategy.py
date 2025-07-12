#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

INPUT_CSV = "sp500_close_prices.csv"
PLOTS_DIR = "plots"
TICKER_A = "AMAT"
TICKER_B = "NXPI"

# Set range of values for window and entry threshold
WINDOW_RANGE = range(20, 101, 10)  # Test windows from 20 to 100 in steps of 10
THRESHOLD_RANGE = np.arange(
    1.0, 3.1, 0.5
)  # Test thresholds from 1.0 to 3.0 in steps of 0.5

EXIT_THRESHOLD = 0.5
COMMISSION = 0.001


def run_optimization(file_path: str, ticker_a: str, ticker_b: str):
    # Load data
    print(f"Loading data for optimization of pair ({ticker_a}, {ticker_b})...")
    try:
        close_prices = pd.read_csv(file_path, index_col=0, parse_dates=True)
        pair_prices = close_prices[[ticker_a, ticker_b]].dropna()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Calculate hedge ratio once
    y = pair_prices[ticker_a]
    x = sm.add_constant(pair_prices[ticker_b])
    model = sm.OLS(y, x).fit()
    hedge_ratio = model.params[1]
    spread = pair_prices[ticker_a] - hedge_ratio * pair_prices[ticker_b]

    results = []

    print(
        f"Optimizing over {len(WINDOW_RANGE)} windows and {len(THRESHOLD_RANGE)} thresholds..."
    )

    # Loop through each combination of parameters
    for window in WINDOW_RANGE:
        for threshold in THRESHOLD_RANGE:
            # Calculate z-score for current window
            rolling_mean = spread.rolling(window=window).mean()
            rolling_std = spread.rolling(window=window).std()
            z_score = (spread - rolling_mean) / rolling_std

            # Generate signals
            long_entry = z_score < -threshold
            short_entry = z_score > threshold
            exit_signal = abs(z_score) < EXIT_THRESHOLD

            positions = pd.Series(np.nan, index=z_score.index)
            positions[long_entry] = 1
            positions[short_entry] = -1
            positions[exit_signal] = 0
            positions.ffill(inplace=True)
            positions.fillna(0, inplace=True)

            # Calculate returns
            daily_returns = pair_prices.pct_change()
            portfolio_returns = positions.shift(1) * (
                daily_returns[ticker_a] - hedge_ratio * daily_returns[ticker_b]
            )
            trades = positions.diff().abs()
            transaction_costs = trades * COMMISSION
            net_portfolio_returns = portfolio_returns - transaction_costs

            # Calculate sharpe ratio
            sharpe_ratio = (
                (net_portfolio_returns.mean() / net_portfolio_returns.std())
                * np.sqrt(252)
                if net_portfolio_returns.std() != 0
                else 0
            )

            results.append(
                {"window": window, "threshold": threshold, "sharpe_ratio": sharpe_ratio}
            )

    # Create DataFrame from results for analysis
    results_df = pd.DataFrame(results)

    # Find best params
    best_params = results_df.loc[results_df["sharpe_ratio"].idxmax()]

    print("\n--- Optimization Complete ---")
    print("Best Parameters:")
    print(best_params)
    print("-----------------------------")

    # Create heatmap of results
    heatmap_data = results_df.pivot(
        index="window", columns="threshold", values="sharpe_ratio"
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
    plt.title(
        f"Sharpe Ratio by Window and Entry Threshold for ({ticker_a}, {ticker_b})"
    )
    plt.xlabel("Entry Z-Score Threshold")
    plt.ylabel("Rolling Window Size")

    # Save plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_filename = os.path.join(
        PLOTS_DIR, f"optimization_heatmap_{ticker_a}_{ticker_b}.png"
    )
    plt.savefig(plot_filename)
    plt.close()

    print(f"Optimization heatmap saved to '{plot_filename}'")


if __name__ == "__main__":
    run_optimization(INPUT_CSV, TICKER_A, TICKER_B)

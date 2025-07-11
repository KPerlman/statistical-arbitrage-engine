#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

INPUT_CSV = "sp500_close_prices.csv"
PLOTS_DIR = "plots"
# Simplify for visualization
SAMPLE_TICKERS = ["AAPL", "MSFT", "JPM", "AMZN", "JNJ", "GOOGL", "XOM", "UNH"]

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Paired")


def analyze_sp500_data(file_path: str):
    # Load data
    print(f"Loading data from '{file_path}'...")
    try:
        # First column in CSV is date so set it as index
        close_prices = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print("Data loaded successfully.")
        print(f"Data shape: {close_prices.shape}")
        print(f"Date range: {close_prices.index.min()} to {close_prices.index.max()}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Check if sample tickers to plot are actually in dataset
    available_tickers = [t for t in SAMPLE_TICKERS if t in close_prices.columns]
    if not available_tickers:
        print(
            "None of the sample tickers are in the dataset. Grabbing a few available ones."
        )
        available_tickers = close_prices.columns[:5].tolist()

    print(f"\nAnalyzing the following sample tickers: {available_tickers}")

    # Calculate daily returns
    daily_returns = close_prices.pct_change().dropna()
    print("\nCalculated daily returns.")

    # Gen visualizations
    print("Generating visualizations...")

    # Plot normalized prices over time (start at 100)
    (
        close_prices[available_tickers] / close_prices[available_tickers].iloc[0] * 100
    ).plot(figsize=(15, 7))
    plt.title("Normalized Stock Prices", fontsize=16)
    plt.ylabel("Normalized Price (Start = 100)")
    plt.xlabel("Date")
    plt.legend(title="Ticker")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sp500_normalized_prices.png"))
    print(f" - Saved 'sp500_normalized_prices.png' to '{PLOTS_DIR}/'")
    plt.close()

    # Plot distribution of daily returns for one stock
    sample_stock = available_tickers[0]
    plt.figure(figsize=(12, 6))
    sns.histplot(daily_returns[sample_stock], bins=100, kde=True)
    plt.title(f"Distribution of Daily Returns for {sample_stock}", fontsize=16)
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{sample_stock}_returns_distribution.png"))
    print(f" - Saved '{sample_stock}_returns_distribution.png' to '{PLOTS_DIR}/'")
    plt.close()

    # Plot correlation matrix heatmap
    correlation_matrix = daily_returns[available_tickers].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Daily Returns", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sp500_correlation_heatmap.png"))
    print(f" - Saved 'sp500_correlation_heatmap.png' to '{PLOTS_DIR}/'")
    plt.close()

    print("\nAnalysis complete.")


if __name__ == "__main__":
    analyze_sp500_data(INPUT_CSV)

#!/usr/bin/env python

import pandas as pd
from statsmodels.tsa.stattools import coint
import itertools

INPUT_CSV = "sp500_close_prices.csv"
P_VALUE_THRESHOLD = 0.05  # Standard threshold for cointegration


def find_cointegrated_pairs(file_path: str):
    # Load data
    print(f"Loading data from '{file_path}'...")
    try:
        close_prices = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print("Data loaded successfully.")
        print(f"Testing for cointegration on {len(close_prices.columns)} stocks...")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    tickers = close_prices.columns.tolist()

    # Test all pairs for cointegration
    print(
        f"Finding cointegrated pairs with a p-value threshold of {P_VALUE_THRESHOLD}..."
    )

    # Create all unique pairs of tickers
    pairs = list(itertools.combinations(tickers, 2))
    print(f"There are {len(pairs)} unique pairs to test.")

    cointegrated_pairs = []

    # Progress indicator
    total_pairs = len(pairs)
    for i, (ticker1, ticker2) in enumerate(pairs):
        # Every 1000 pairs print progress
        if (i + 1) % 1000 == 0:
            print(f"  ... tested {i + 1}/{total_pairs} pairs")

        # Engle-Granger cointegration test
        # Returns cointegration test statistic, p-value, and critical values
        # Only p-value is relevant here
        try:
            score, pvalue, _ = coint(close_prices[ticker1], close_prices[ticker2])
            if pvalue < P_VALUE_THRESHOLD:  # Cointegrated
                pair_info = {
                    "pair": (ticker1, ticker2),
                    "p_value": pvalue,
                    "score": score,
                }
                cointegrated_pairs.append(pair_info)
        except Exception as e:
            continue  # Ignore errors

    print(f"\nTesting complete. Found {len(cointegrated_pairs)} cointegrated pairs.")

    # Output
    if cointegrated_pairs:
        # Sort pairs by p-value where lower is better
        sorted_pairs = sorted(cointegrated_pairs, key=lambda x: x["p_value"])

        print("\nTop 15 Cointegrated Pairs (sorted by p-value):")
        print("-" * 50)
        print(f"{'Pair':<18} | {'p-value':<12}")
        print("-" * 50)
        for pair_info in sorted_pairs[:15]:
            print(f"{str(pair_info['pair']):<18} | {pair_info['p_value']:.6f}")
        print("-" * 50)

        # Save pairs to CSV
        pd.DataFrame(sorted_pairs).to_csv("cointegrated_pairs.csv", index=False)
        print("\nFull list of cointegrated pairs saved to 'cointegrated_pairs.csv'.")
    else:
        print("\nNo cointegrated pairs found with the current threshold.")


if __name__ == "__main__":
    find_cointegrated_pairs(INPUT_CSV)

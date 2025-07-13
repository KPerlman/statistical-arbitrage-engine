# Statistical Arbitrage Engine

### About this Project

This is a project I'm working on to apply some of the signal processing and data analysis skills from my ECE background to quantitative finance. I'm building a set of Python scripts to see if I can systematically find and test mean-reversion trading strategies.

The core idea is to:

1. Pull historical price data for a universe of stocks (like the S&P 500).

2. Find pairs of stocks whose prices tend to move together (cointegrated).

3. Model the spread between a pair's prices.

4. Test a strategy that trades on the assumption that the spread will revert to its historical mean.

I'm mostly using this to learn more about time-series analysis, financial data, and what it takes to test a trading idea from scratch.

### Tech Stack

* **Language:** Python 3

* **Libraries:** pandas, yfinance, statsmodels, matplotlib, seaborn, pykalman

### Project Structure

The project is a pipeline of scripts, each handling one part of the process:

1. `01_data_collection.py`: Fetches historical close prices for all S&P 500 tickers from Yahoo Finance and saves the data to a CSV.

2. `02_analyze_data.py`: Runs some basic exploratory data analysis (EDA) on the price data and generates a few plots (e.g., correlation matrix).

3. `03_find_cointegrated_pairs.py`: Iterates through all possible pairs of stocks and runs an Engle-Granger cointegration test to find statistically significant pairs. The results are ranked by p-value.

4. `04_plot_spread.py`: Takes a specific pair (e.g., the top result from the previous script) and plots its price spread and rolling z-score over time to visualize the mean-reversion behavior.

5. `05_backtest_strategy.py`: A backtester that runs a trading simulation on the top N cointegrated pairs. It includes a simple transaction cost model and calculates key performance metrics (CAGR, Sharpe Ratio, Max Drawdown) for each pair.

6. `06_optimize_strategy.py`: Takes the best-performing pair and runs a grid search to find the optimal lookback window and z-score threshold. It also outputs a heatmap based on the results.

7. `07_kalman_filter_backtest.py`: Implements a more advanced backtest using a Kalman Filter to dynamically calculate the hedge ratio over time.
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

* **Libraries:** pandas, yfinance, statsmodels, matplotlib, seaborn

### Project Structure

The project is a pipeline of scripts, each handling one part of the process:

1. `01_data_collection.py`: Fetches historical close prices for all S&P 500 tickers from Yahoo Finance and saves the clean data to a CSV.

2. `02_analyze_data.py`: Runs some basic exploratory data analysis (EDA) on the price data and generates a few plots (e.g., correlation matrix).

3. `03_find_cointegrated_pairs.py`: Iterates through all possible pairs of stocks and runs an Engle-Granger cointegration test to find statistically significant pairs. The results are ranked by p-value.

4. `04_plot_spread.py`: Takes a specific pair (e.g., the top result from the previous script) and plots its price spread and rolling z-score over time to visualize the mean-reversion behavior.

5. `05_backtest_strategy.py`: Simulates a simple trading strategy on a given pair based on z-score entry/exit thresholds.

### Next Steps & To-Do

This is still a work in progress. Here's what I'm planning to work on next:

* [x] ~~Build an event-driven backtester to simulate performance.~~
* \[ \] **Improve the Backtester:**

  * \[ \] Add a more realistic cost model (commissions, slippage).

  * \[ \] Implement a proper portfolio manager to track equity and P&L.

* \[ \] **Add More Performance Metrics:**

  * \[ \] Calculate Sharpe Ratio, Max Drawdown, CAGR, etc.

  * \[ \] Generate a tear sheet for backtest results.

* \[ \] **(Future Idea) Dynamic Modeling:**

  * \[ \] Look into using a Kalman Filter to dynamically update the hedge ratio instead of using a static one from the initial regression. This should be a better model for a relationship that changes over time.
# Project Conclusion

### Project Goal

The goal of this project was to build a Python-based engine to explore a statistical arbitrage pairs trading strategy. This involved screening S&P 500 stocks to find cointegrated pairs, and then building and testing a trading model based on the mean-reversion of their price spread.

### Summary of Findings

This strategy identified several promising pairs. The top candidate, **Applied Materials (AMAT) and NXP Semiconductors (NXPI)**, had a very strong statistical relationship (p-value < 0.00001).

I ended up testing two different models on this pair:

1.  **A Static Model:** This model used a more simplistic linear regression over the historical dataset to calculate a fixed hedge ratio. The strategy parameters (lookback window and z-score entry threshold) were optimized via a grid search.

2.  **A Dynamic Model:** This model used a Kalman Filter to continuously update the hedge ratio at each time step, meaning the model would adapt to changes in the relationship between the stocks.

### Performance Comparison

After running backtests for both models (including a 0.1% commission per trade), the results for the `(AMAT, NXPI)` pair were as follows:

| Metric             | Static Model (Optimized) | Kalman Filter Model |
| ------------------ | ------------------------ | ------------------- |
| **CAGR** | **29.55%** | 17.59%              |
| **Sharpe Ratio** | **1.38** | 0.71                |
| **Max Drawdown** | **-32.25%** | -33.53%             |
| **Total Trades** | 81                       | 47                  |

### Key Takeaway: Simplicity Can Win

The most interesting result of this project was that the **simpler, static model significantly outperformed the more complex Kalman Filter model** for this specific pair and time period.

While the Kalman Filter is in theory a more sophisticated approach, its constant adjustments may have been overfitting to short-term noise in the price data. The static model, relying on a stable, long-term average, proved to be more robust and generated better risk-adjusted returns.

More complexity does not always guarantee better performance. The best model can easily be the simplest one that effectively captures the underlying phenomenon.
# Statistical Arbitrage Engine

![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-blue)

## Idea

This repository contains the R&D of an engine for discovering, modeling, and backtesting mean-reversion trading strategies. The goal is to build a robust, end-to-end framework that moves beyond a simple case study into a scalable and systematic research tool.

Currently in the planning and initial development phase.

---

## Tech Stack

* **Language:** Python
* **Core Libraries:** Pandas, NumPy, Statsmodels, Matplotlib

---

## Project Roadmap

### Phase 1: Systematic Pair Discovery
Build a scalable script to screen the S&P 500 and identify cointegrated pairs using rigorous statistical tests. Essentially, create a ranked list of the most promising trading candidates.

### Phase 2: Backtesting Framework
Develop an event-driven backtester to simulate strategy performance. Include a cost model for transaction fees and slippage, plus a risk management layer to exit trades if the statistical relationship weakens.

### Phase 3: Performance Analysis & Visualization
Implement functions to calculate key performance metrics (Sharpe Ratio, Max Drawdown, CAGR) and create visualizations to clearly display strategy performance and trade behavior.

### Phase 4 (Potentially): Dynamic Modeling
Upgrade the standard static model with a Kalman Filter to dynamically model the pair's hedge ratio and relationship over time.

---

## Planned Features

* Systematic discovery of trading pairs across an entire index.
* Realistic backtesting with transaction costs and risk management.
* Dynamic modeling of asset relationships using Kalman Filters.
* Detailed performance analytics and data visualization.
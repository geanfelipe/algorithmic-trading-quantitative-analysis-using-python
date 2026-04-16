# Quant Developer Portfolio: Algorithmic Trading and Portfolio Analytics in Python

This repository is a practical quant research portfolio project focused on building and evaluating trading analytics in Python. It demonstrates applied skills in return modeling, risk estimation, technical indicators, and portfolio construction using real market data.

## Portfolio Snapshot

- **Role target:** Quant Developer / Quant Research Engineer
- **Domain:** Equities, technical analysis, portfolio analytics
- **Stack:** Python, pandas, NumPy, SciPy, statsmodels, matplotlib, yfinance
- **Data:** Historical daily OHLCV equity/index data in the `data/` folder

## Learning Objectives (Udemy Track)

This project is also my hands-on implementation of the core outcomes from my quantitative finance and Python training.

- Calculate stock returns manually and with Python using real-world market data from free sources
- Work extensively with key Python libraries for quant analysis: pandas, NumPy, SciPy, and matplotlib
- Build intuition for the mathematics behind financial models, with transparent step-by-step implementations
- Demonstrate diversification and show how portfolio risk can be lower than the risk of individual assets
- Estimate expected stock returns using:
	- Mean return method
	- State-contingent weighted probabilities
	- Asset pricing model approaches
- Calculate and interpret total risk, market risk, and firm-specific risk
- Measure portfolio performance using return and risk metrics
- Optimize portfolios by balancing return maximization and risk minimization
- Create custom Python functions to automate investment analysis and portfolio management workflows
- Rebuild selected computations from scratch to understand the mechanics behind library-based implementations

## What This Project Demonstrates

- Building reusable quantitative utility functions for return and volatility analysis
- Estimating market risk metrics (covariance-based beta and regression-based beta)
- Implementing technical indicators (MACD, ATR, slope, Renko)
- Comparing rebalanced vs buy-and-hold portfolio behavior
- Computing strategy performance metrics: CAGR, volatility, Sharpe ratio, max drawdown
- Exploring multi-asset combinations and ranking by risk-adjusted return

## Project Structure

- `portfolio_analysis.py`
	- Return calculation and risk analytics for a stock vs benchmark index
	- Beta estimation using both manual covariance/variance and linear regression

- `technical_analysis_indicator.py`
	- Technical-indicator and strategy helper functions
	- Includes MACD, ATR, slope estimation, Renko transformation, and portfolio-level metrics
	- Includes combination search logic for selecting high-performing ticker baskets

- `rebalacing_test.py`
	- Rebalancing comparison experiment (periodic rebalance vs one-time purchase)

- `data/*.csv`
	- Daily historical datasets used for local analysis and reproducibility

## Quant Methods Implemented

### Risk and Return Analytics

- Simple returns using percentage change
- Rolling moving averages on return series
- Annualized standard deviation from daily variance
- Beta estimation:
	- Manual method: $\beta = \frac{\operatorname{Cov}(R_i, R_m)}{\operatorname{Var}(R_m)}$
	- Regression method via `scipy.stats.linregress`

### Strategy Evaluation Metrics

- CAGR
- Annualized volatility
- Sharpe ratio (configurable risk-free rate)
- Maximum drawdown

### Technical Indicators

- MACD and signal line
- ATR and true range components
- Rolling slope (angle form)
- Renko brick conversion and trend encoding

## Environment and Setup

### 1) Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

## How To Run

Run each script independently based on the analysis you want:

```powershell
python portfolio_analysis.py
python technical_analysis_indicator.py
python rebalacing_test.py
```

Plots will open using matplotlib for visual inspection of strategy and portfolio behavior.

## Recruiter-Focused Highlights

- Translates quantitative finance concepts into production-style Python workflows
- Uses both formula-driven and library-driven implementations to cross-validate outputs
- Applies risk-adjusted performance thinking, not only raw return optimization
- Shows ability to structure research code into modular analytical components

## Potential Next Enhancements

- Add unit tests for financial metric functions and edge cases
- Introduce transaction costs/slippage in strategy evaluation
- Add walk-forward validation and out-of-sample evaluation framework
- Export results to report tables for reproducible recruiter demos
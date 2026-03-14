#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 20:41:28 2026

@author: geanfelipe
"""

# import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# from alpha_vantage.timeseries import TimeSeries

amzn_df = pd.read_csv("data/amzn_us_d.csv")
amzn_df.drop(columns=["Open", "High", "Low", "Volume"], inplace=True)
amzn_df["Date"] = pd.to_datetime(amzn_df["Date"])
print(amzn_df.dtypes)
amzn_df.set_index("Date", inplace=True)
a = 12

aapl_df = pd.read_csv("data/aapl_us_d.csv")
aapl_df.drop(columns=["Open", "High", "Low", "Volume"], inplace=True)
aapl_df["Date"] = pd.to_datetime(aapl_df["Date"])
print(aapl_df.dtypes)
aapl_df.set_index("Date", inplace=True)

# pct_change is equals to current value - previous value/prevous value
amzn_daily_return = amzn_df.pct_change()
aapl_daily_return = aapl_df.pct_change()


(1 + amzn_daily_return).cumprod().plot(title="AMZN daily simple return")
(1 + aapl_daily_return).cumprod().plot(title="AAPL daily simple return")

fig, ax = plt.subplots()
plt.plot((1 + amzn_daily_return).cumprod())
plt.plot((1 + aapl_daily_return).cumprod())
plt.title("AMZN vs AAPL")
plt.ylabel("Cumulative return")
plt.xlabel("Daily")
ax.legend(["AMZN return", "AAPL return"])


# =============================================================================
# rebalancing portfolio
# =============================================================================
portfolio_value = 10000
returns = pd.concat([aapl_daily_return, amzn_daily_return], axis=1)
returns.columns = ["AAPL", "AMZN"]
portfolio_weight = np.array([0.5, 0.5])
rebalansed_portfolio_returns = returns.dot(portfolio_weight)
rebalansed_portfolio_cum = (
    1 + rebalansed_portfolio_returns
).cumprod() * portfolio_value


# =============================================================================
# one time purchase
# =============================================================================
amzn_s = 5000 / amzn_df.Close.iloc[0]
aapl_s = 5000 / aapl_df.Close.iloc[0]
closes = pd.concat([aapl_df["Close"], amzn_df["Close"]], axis=1)
closes.columns = ["AAPL", "AMZN"]
portfolio_shares = np.array([amzn_s, aapl_s])
one_time_purchased_portfolio_returns = closes.dot(portfolio_shares)

# =============================================================================
# now let's validate if when one time purchased case throughout the horizon
# remains the same
# =============================================================================
portfolio_value_at_the_end = (
    closes.iloc[-1, 0] * amzn_s + closes.iloc[-1, 1] * aapl_s
)
amzn_weight = closes.iloc[-1, 0] * amzn_s / portfolio_value_at_the_end
aapl_weight = closes.iloc[-1, 1] * aapl_s / portfolio_value_at_the_end

amzn_return_acum = (1 + returns).cumprod()["AMZN"].iloc[-1]
aapl_return_acum = (1 + returns).cumprod()["AAPL"].iloc[-1]
aapl_final_value = aapl_return_acum * closes.iloc[0, 0] * 5000
amzn_final_value = amzn_return_acum * closes.iloc[0, 1] * 5000

plt.plot(rebalansed_portfolio_cum)
plt.plot(one_time_purchased_portfolio_returns)
plt.legend(["Portfolio Rebalanced", "One time purchased"])
plt.ylabel("Cumulative return")
plt.xlabel("Date")


randomm_returns = np.random.normal(0, 0.01, 1000)
np.random.nor
price = 100 * np.cumprod(1 + randomm_returns)

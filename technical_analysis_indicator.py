#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 22:13:01 2026

@author: geanfelipe
"""

import pandas
from pathlib import Path
from twelvedata import TDClient
import numpy
from datetime import datetime, timezone, timedelta
import copy
import yfinance
import matplotlib.pyplot
import statsmodels.api as statsmodel
from stocktrends import Renko
import itertools


from dotenv import load_dotenv
import os

load_dotenv()  # loads variables from .env
MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY")


def macd(
    quotes: pandas.DataFrame,
    ema_fast_period: int = 12,
    ema_slow_period: int = 26,
    signal: int = 9,
) -> pandas.DataFrame:
    """Calculate MACD for multiple tickers"""

    results = []

    close_prices = quotes["Close"]

    for ticker in close_prices.columns:
        price = close_prices[ticker]

        ema_fast = price.ewm(
            span=ema_fast_period, min_periods=ema_fast_period
        ).mean()
        ema_slow = price.ewm(
            span=ema_slow_period, min_periods=ema_slow_period
        ).mean()

        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, min_periods=signal).mean()

        temp = pandas.DataFrame(
            {(ticker, "MACD"): macd, (ticker, "Signal"): signal_line}
        )
        temp.dropna(inplace=True)
        results.append(temp)

    macd_df = pandas.concat(results, axis=1)
    macd_df.columns = pandas.MultiIndex.from_tuples(macd_df.columns)

    return macd_df


def atr(quotes: pandas.DataFrame, number_of_periods: int) -> pandas.DataFrame:
    "function to caculate True Range and Average True Range"
    results = []

    for ticker in quotes.columns.get_level_values(1).unique():
        df = pandas.DataFrame()
        df["H-L"] = abs(quotes["High"][ticker] - quotes["Low"][ticker])
        df["H-PC"] = abs(
            quotes["High"][ticker] - quotes["Close"][ticker].shift(1)
        )
        df["L-PC"] = abs(
            quotes["Low"][ticker] - quotes["Close"][ticker].shift(1)
        )
        df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1, skipna=False)
        df["ATR"] = df["TR"].rolling(number_of_periods).mean()
        df.drop(["H-L", "H-PC", "L-PC"], axis=1, inplace=True)
        temp = pandas.DataFrame(
            {(ticker, "TR"): df["TR"], (ticker, "ATR"): df["ATR"]}
        )
        results.append(temp)
    atr_df = pandas.concat(results, axis=1)
    atr_df.columns = pandas.MultiIndex.from_tuples(atr_df.columns)
    return atr_df


def slope(serie: pandas.Series, window: int) -> numpy.array:
    "function to calculate the slope of N consecutive points on a plot"
    slopes = [i * 0 for i in range(window - 1)]
    for i in range(window, len(serie) + 1):
        y = serie[i - window : i]
        x = numpy.array(range(window))
        y_range = y.max() - y.min()
        if y_range == 0:
            slopes.append(0)
            continue
        y_scaled = (y - y.min()) / y_range
        x_scaled = (x - x.min()) / (x.max() - x.min())
        x_scaled = statsmodel.add_constant(x_scaled)
        model = statsmodel.OLS(y_scaled, x_scaled)
        results = model.fit()
        slopes.append(results.params.iloc[-1])
    slope_angle = numpy.rad2deg(numpy.arctan(numpy.array(slopes)))
    return numpy.array(slope_angle)


def renko(quotes: pandas.DataFrame) -> dict:
    "function to convert quotes data into renko bricks"
    atr_df = atr(quotes, 120)
    results = {}

    for ticker in quotes.columns.get_level_values(1).unique():
        ticker_quotes = quotes.xs(ticker, level=1, axis=1).copy()
        ticker_quotes = ticker_quotes.reset_index()
        date_column = ticker_quotes.columns[0]

        ticker_df = pandas.DataFrame(
            {
                "date": pandas.to_datetime(ticker_quotes[date_column]),
                "open": ticker_quotes["Open"],
                "high": ticker_quotes["High"],
                "low": ticker_quotes["Low"],
                "close": ticker_quotes["Close"],
                "volume": ticker_quotes["Volume"],
            }
        )

        renko = Renko(ticker_df)
        renko.brick_size = max(0.5, round(atr_df[ticker]["ATR"].iloc[-1]))
        renko_df = renko.get_ohlc_data()
        renko_df["bar_num"] = numpy.where(
            renko_df["uptrend"] == True,
            1,
            numpy.where(renko_df["uptrend"] == False, -1, 0),
        )
        for i in range(1, len(renko_df["bar_num"])):
            curr_bar = renko_df["bar_num"].iloc[i]
            prev_bar = renko_df["bar_num"].iloc[i - 1]
            if curr_bar > 0 and prev_bar > 0:
                renko_df.loc[i, "bar_num"] = curr_bar + prev_bar
            elif curr_bar < 0 and prev_bar < 0:
                renko_df.loc[i, "bar_num"] = curr_bar + prev_bar
        renko_df.drop_duplicates(subset="date", keep="last", inplace=True)
        results[ticker] = renko_df
    return results


def cagr(
    ticket_return: pandas.DataFrame, quotes_by_day: int
) -> pandas.DataFrame:
    "function to calculaat ethe Cumulative Annual Growth of a trading strategy"
    ticket_return_copy = ticket_return.copy()
    ticket_return_copy["cum_return"] = (
        1 + ticket_return_copy["ret"]
    ).cumprod()
    years = len(ticket_return_copy) / (252 * quotes_by_day)
    cagr_df = ticket_return_copy["cum_return"].tolist()[-1] ** (1 / years) - 1

    return cagr_df


def volatility(
    ticket_return: pandas.DataFrame, quotes_by_day: int
) -> pandas.DataFrame:
    "function to calculate annualized volatility of a trading strategy"
    vol = ticket_return["ret"].std() * numpy.sqrt(252 * quotes_by_day)
    return vol


def sharpe(
    ticket_return: pandas.DataFrame, quotes_by_day: int, risk_free_rate: float
) -> pandas.DataFrame:
    "function to calculate sharpe ratio"
    sharpe = (
        cagr(ticket_return, quotes_by_day) - risk_free_rate
    ) / volatility(ticket_return, quotes_by_day)
    return sharpe


def maximum_drawdown(ticket_return: pandas.DataFrame) -> pandas.DataFrame:
    "function to calculate maximum drawdown"
    ticket_return_copy = ticket_return.copy()
    ticket_return_copy["cum_return"] = (
        1 + ticket_return_copy["ret"]
    ).cumprod()
    ticket_return_copy["cum_roll_max"] = ticket_return_copy[
        "cum_return"
    ].cummax()
    ticket_return_copy["drawdown"] = (
        ticket_return_copy["cum_roll_max"] - ticket_return_copy["cum_return"]
    )
    ticket_return_copy["drawdown_pct"] = (
        ticket_return_copy["drawdown"] / ticket_return_copy["cum_roll_max"]
    )
    max_dd = ticket_return_copy["drawdown_pct"].max()
    return max_dd


def find_best_10_stock_combinations(
    ohlc_renko: dict,
    combo_size: int = 10,
    top_n: int = 10,
    quotes_by_day: int = 1,
    risk_free_rate: float = 0.04,
    max_combinations: int | None = None,
) -> pandas.DataFrame:
    """Evaluate stock combinations using the current ohlc_renko structure.

    Parameters are tailored to this project where `ohlc_renko` is a dict in the
    format: {ticker: DataFrame with columns ['Date', 'ret', ...]}.
    """
    if combo_size <= 0:
        raise ValueError("combo_size must be positive")

    returns_by_ticker = {}
    for ticker, ticker_df in ohlc_renko.items():
        if "Date" not in ticker_df.columns or "ret" not in ticker_df.columns:
            continue
        returns_by_ticker[ticker] = ticker_df.set_index("Date")["ret"]

    if len(returns_by_ticker) < combo_size:
        raise ValueError(
            "Not enough tickers with return data: "
            f"{len(returns_by_ticker)} available, {combo_size} required"
        )

    returns_df = pandas.DataFrame(returns_by_ticker).dropna(how="all")
    tickers = sorted(returns_df.columns.tolist())

    combinations_iter = itertools.combinations(tickers, combo_size)
    if max_combinations is not None:
        combinations_iter = itertools.islice(
            combinations_iter, max_combinations
        )

    rows = []
    for combo in combinations_iter:
        combo_returns = returns_df[list(combo)].mean(axis=1).dropna()
        if combo_returns.empty:
            continue

        combo_df = pandas.DataFrame({"ret": combo_returns})
        vol = volatility(combo_df, quotes_by_day)

        if vol == 0 or pandas.isna(vol):
            continue

        cagr_value = cagr(combo_df, quotes_by_day)
        sharpe_value = sharpe(combo_df, quotes_by_day, risk_free_rate)
        max_dd_value = maximum_drawdown(combo_df)

        rows.append(
            {
                "tickers": ",".join(combo),
                "cagr": cagr_value,
                "volatility": vol,
                "sharpe": sharpe_value,
                "max_drawdown": max_dd_value,
                "score": sharpe_value - max_dd_value,
            }
        )

    if not rows:
        return pandas.DataFrame(
            columns=[
                "tickers",
                "cagr",
                "volatility",
                "sharpe",
                "max_drawdown",
                "score",
            ]
        )

    results = pandas.DataFrame(rows)
    results.sort_values(
        ["score", "sharpe", "cagr"], ascending=False, inplace=True
    )
    return results.head(top_n).reset_index(drop=True)


def load_quotes(tickers, data_path="data/"):
    """
    Try to download quotes from Yahoo.
    If it fails (rate limit), load local CSV files instead.
    """

    try:
        print("Downloading data from Yahoo Finance...")
        quotes = yfinance.download(
            tickers,
            period="60d",
            interval="5m",
            group_by="column",
            auto_adjust=False,
            threads=True,
        )

        if quotes.empty:
            raise Exception("Yahoo returned empty dataframe")

        print("Yahoo download successful")
        return quotes

    except Exception as e:
        print("Yahoo download failed. Loading local CSV files.")
        print(e)

        dfs = []

        for ticker in tickers:
            file = Path(data_path) / f"{ticker.lower()}_us_d.csv"

            if not file.exists():
                print(f"File not found for {ticker}")
                continue

            temp = pandas.read_csv(file)
            temp.rename(columns={"Date": "Datetime"}, inplace=True)

            # Convert date
            temp["Datetime"] = pandas.to_datetime(temp["Datetime"])

            # Set time = 00:00:00
            temp["Datetime"] = temp["Datetime"].dt.normalize()

            temp.set_index("Datetime", inplace=True)

            # Build MultiIndex columns
            temp.columns = pandas.MultiIndex.from_product(
                [temp.columns, [ticker]]
            )

            dfs.append(temp)

        quotes = pandas.concat(dfs, axis=1)
        return quotes


if __name__ == "__main__":
    tickers = [
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "AMZN",  # Amazon
        "GOOGL",  # Alphabet
        "META",  # Meta Platforms
        "NVDA",  # Nvidia
        "TSLA",  # Tesla
        "JPM",  # JPMorgan Chase
        "V",  # Visa
        "MA",  # Mastercard
        "UNH",  # UnitedHealth
        "HD",  # Home Depot
        "PG",  # Procter & Gamble
        "KO",  # Coca-Cola
        "PEP",  # PepsiCo
        "COST",  # Costco
        "AVGO",  # Broadcom
        "AMD",  # Advanced Micro Devices
        "CRM",  # Salesforce
        "ADBE",  # Adobe
    ]

    quotes = load_quotes(tickers)
    available_tickers = quotes.columns.get_level_values(1).unique().tolist()
    macd_data = macd(quotes)
    renko_data = renko(quotes)
    ohlc_renko = {}
    tickers_signal = {}
    tickers_return = {}

    for ticker in available_tickers:
        print(f"merging for {ticker}")

        ticker_ohlc = pandas.DataFrame(
            {
                "Date": quotes["Close"][ticker].index,
                "Open": quotes["Open"][ticker].values,
                "High": quotes["High"][ticker].values,
                "Low": quotes["Low"][ticker].values,
                "Close": quotes["Close"][ticker].values,
                "Adj Close": quotes["Close"][ticker].values,
                "Volume": quotes["Volume"][ticker].values,
            }
        )

        ticker_ohlc.dropna(inplace=True)
        ticker_ohlc["Date"] = pandas.to_datetime(ticker_ohlc["Date"])

        ticker_renko = renko_data[ticker].copy()
        ticker_renko = ticker_renko[["date", "bar_num"]]
        ticker_renko.rename(columns={"date": "Date"}, inplace=True)
        ticker_renko["Date"] = pandas.to_datetime(ticker_renko["Date"])

        merged = ticker_ohlc.merge(ticker_renko, how="outer", on="Date")
        merged.sort_values("Date", inplace=True)
        merged["bar_num"] = merged["bar_num"].ffill()
        merged["bar_num"] = merged["bar_num"].fillna(0)

        merged_macd = pandas.DataFrame(
            {
                "Date": macd_data.index,
                "macd": macd_data[(ticker, "MACD")].values,
                "macd_sig": macd_data[(ticker, "Signal")].values,
            }
        )

        merged = merged.merge(merged_macd, how="left", on="Date")
        merged["macd"] = merged["macd"].ffill()
        merged["macd_sig"] = merged["macd_sig"].ffill()
        merged["macd_slope"] = slope(merged["macd"].fillna(0), 5)
        merged["macd_sig_slope"] = slope(merged["macd_sig"].fillna(0), 5)

        ohlc_renko[ticker] = merged.reset_index(drop=True)

    for ticker in available_tickers:
        print("calculating daily returns for", ticker)

        tickers_signal[ticker] = ""
        tickers_return[ticker] = []

        for i in range(len(ohlc_renko[ticker])):
            if tickers_signal[ticker] == "":
                tickers_return[ticker].append(0)
                if i > 0:
                    if (
                        ohlc_renko[ticker]["bar_num"].iloc[i] >= 2
                        and ohlc_renko[ticker]["macd"].iloc[i]
                        > ohlc_renko[ticker]["macd_sig"].iloc[i]
                        and ohlc_renko[ticker]["macd_slope"].iloc[i]
                        > ohlc_renko[ticker]["macd_sig_slope"].iloc[i]
                    ):
                        tickers_signal[ticker] = "Buy"
                    elif (
                        ohlc_renko[ticker]["bar_num"].iloc[i] <= -2
                        and ohlc_renko[ticker]["macd"].iloc[i]
                        < ohlc_renko[ticker]["macd_sig"].iloc[i]
                        and ohlc_renko[ticker]["macd_slope"].iloc[i]
                        < ohlc_renko[ticker]["macd_sig_slope"].iloc[i]
                    ):
                        tickers_signal[ticker] = "Sell"

            elif tickers_signal[ticker] == "Buy":
                tickers_return[ticker].append(
                    ohlc_renko[ticker]["Adj Close"].iloc[i]
                    / ohlc_renko[ticker]["Adj Close"].iloc[i - 1]
                    - 1
                )
                if i > 0:
                    if (
                        ohlc_renko[ticker]["bar_num"].iloc[i] <= -2
                        and ohlc_renko[ticker]["macd"].iloc[i]
                        < ohlc_renko[ticker]["macd_sig"].iloc[i]
                        and ohlc_renko[ticker]["macd_slope"].iloc[i]
                        < ohlc_renko[ticker]["macd_sig_slope"].iloc[i]
                    ):
                        tickers_signal[ticker] = "Sell"
                    elif (
                        ohlc_renko[ticker]["macd"].iloc[i]
                        < ohlc_renko[ticker]["macd_sig"].iloc[i]
                        and ohlc_renko[ticker]["macd_slope"].iloc[i]
                        < ohlc_renko[ticker]["macd_sig_slope"].iloc[i]
                    ):
                        tickers_signal[ticker] = ""

            elif tickers_signal[ticker] == "Sell":
                tickers_return[ticker].append(
                    ohlc_renko[ticker]["Adj Close"].iloc[i - 1]
                    / ohlc_renko[ticker]["Adj Close"].iloc[i]
                    - 1
                )
                if i > 0:
                    if (
                        ohlc_renko[ticker]["bar_num"].iloc[i] >= 2
                        and ohlc_renko[ticker]["macd"].iloc[i]
                        > ohlc_renko[ticker]["macd_sig"].iloc[i]
                        and ohlc_renko[ticker]["macd_slope"].iloc[i]
                        > ohlc_renko[ticker]["macd_sig_slope"].iloc[i]
                    ):
                        tickers_signal[ticker] = "Buy"
                    elif (
                        ohlc_renko[ticker]["macd"].iloc[i]
                        > ohlc_renko[ticker]["macd_sig"].iloc[i]
                        and ohlc_renko[ticker]["macd_slope"].iloc[i]
                        > ohlc_renko[ticker]["macd_sig_slope"].iloc[i]
                    ):
                        tickers_signal[ticker] = ""

        ohlc_renko[ticker]["ret"] = numpy.array(tickers_return[ticker])

    strategy_df = pandas.DataFrame(
        {
            ticker: ohlc_renko[ticker].set_index("Date")["ret"]
            for ticker in available_tickers
        }
    )
    strategy_df["ret"] = strategy_df.mean(axis=1)
    strategy_df.dropna(inplace=True)

    print(
        "Strategy CAGR:",
        cagr(
            strategy_df[["ret"]],
            quotes_by_day=quotes.index.normalize().value_counts().max(),
        ),
    )
    print(
        "Strategy Sharpe:",
        sharpe(
            strategy_df[["ret"]],
            quotes_by_day=quotes.index.normalize().value_counts().max(),
            risk_free_rate=0.04,
        ),
    )
    print("Strategy Max Drawdown:", maximum_drawdown(strategy_df[["ret"]]))

    # =============================================================================
    # Equity curve (strategy growth)
    # =============================================================================
    strategy_df["cum_return"] = (1 + strategy_df["ret"]).cumprod()
    strategy_df["cum_return"].plot(
        title="Strategy Equity Curve", figsize=(12, 5)
    )
    matplotlib.pyplot.ylabel("Growth of $1")
    matplotlib.pyplot.show()

    # =============================================================================
    #     Drawdown curve
    # =============================================================================
    strategy_df["roll_max"] = strategy_df["cum_return"].cummax()
    strategy_df["drawdown_pct"] = (
        strategy_df["cum_return"] / strategy_df["roll_max"] - 1
    )
    strategy_df["drawdown_pct"].plot(
        title="Strategy Drawdown (%)", figsize=(12, 4)
    )
    matplotlib.pyplot.show()

    # =============================================================================
    #     Distribution of daily returns
    # =============================================================================
    strategy_df["ret"].hist(bins=80, figsize=(10, 4))
    matplotlib.pyplot.title("Distribution of Daily Strategy Returns")
    matplotlib.pyplot.show()

    # =============================================================================
    #     best 10 stocks
    # =============================================================================

    best_combinations = find_best_10_stock_combinations(
        ohlc_renko,
        combo_size=min(10, len(available_tickers)),
        top_n=10,
        quotes_by_day=1,
        risk_free_rate=0.04,
    )
    print("Top stock combinations:")
    print(best_combinations)

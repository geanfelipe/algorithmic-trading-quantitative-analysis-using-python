from statistics import variance

import pandas
import matplotlib
from pandas import DataFrame
import numpy
from scipy.stats import linregress

stock_data: DataFrame = pandas.read_csv('data/aapl_us_d.csv')
stock_data.rename(columns={'Date': 'date','Close': 'price_t'}, inplace=True)
stock_data = stock_data.set_index('date')

sp_index = pandas.read_csv('data/spx_d.csv')
sp_index.rename(columns={'Date': 'date','Close': 'price_t'}, inplace=True)
sp_index = sp_index.set_index('date')

def calculate_returns(df: DataFrame) -> DataFrame:
    df['return'] = df['price_t'].pct_change(1)
    df['manual_return'] = (df['price_t'] / df['price_t'].shift(1))-1
    df = df[['price_t', 'return']]

    df['return_moving_average_7d'] = df['return'].rolling(min_periods=3,window=7).mean()
    df.dropna(inplace=True)
    return df

def calculate_annualized_standard_deviation(df: DataFrame) -> float:
    ##return df['return'].std()
    df['deviation'] = df['return'] - df['return'].mean()
    df['squared_deviation'] = df['deviation'] ** 2
    daily_variance = df['squared_deviation'].sum() / (len(df.dropna()) - 1)
    return numpy.sqrt(daily_variance) * numpy.sqrt(250)

## estimating market risk of a stock
## this function could be implemented using 

## estimating market risk of a stock using linear regression
# built-in method to get beta: beta, *_ = linregress(x=sp_index['return'], y=stock_data['return'])
def calculate_stock_beta(stock_df: DataFrame, market_index_df: DataFrame) -> float:
    market_index_daily_variance = market_index_df['squared_deviation'].sum() / (len(market_index_df.dropna()) - 1)
    
    product_deviation = market_index_df['deviation'] * stock_df['deviation']
    covariance = product_deviation.sum() / (len(product_deviation.dropna()) - 1)
    
    beta = covariance / market_index_daily_variance
    
    return beta
    

stock_data = calculate_returns(stock_data)
sp_index = calculate_returns(sp_index)
stock_std = calculate_annualized_standard_deviation(stock_data)
market_index_std = calculate_annualized_standard_deviation(sp_index)

stock_beta = calculate_stock_beta(stock_data, sp_index)
beta, *_ = linregress(x=sp_index['return'], y=stock_data['return'])

annualized_mean_return = (1+stock_data['return'].mean())**250 - 1
stock_data['price_t'].plot(figsize=(12,8))
stock_data['return'].plot(figsize=(12,8))
stock_data[['return', 'return_moving_average_7d']].plot(figsize=(12,8))



from scipy.stats import norm
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def read_csv(file_path, tick):
    """
    Reads a file_path for a gvin ticker, and returns a dataframe with date and closing price 
    @params:
        file_path: path to the destination of the file
        tick: ticker of the stock

    @returns:
        temp: pandas dataframe with closing price and date of the stock

    """
    temp = pd.read_csv(file_path, parse_dates=[0], index_col=0)
    temp.rename(columns={"Close" : tick})
    temp.drop(["Open", "High", "Low", "Adj Close", "Volume"], axis = 1, inplace = True)
    return temp


def split_timeseries(timeSeries):
    """
    @params:

    @returns:

    Splits a dataframe into a training set containing 80% of the data, and a testing set containing 
    """
    Train = timeSeries[0 : round(len(timeSeries)*0.5)]
    Test = timeSeries[round(len(timeSeries)*0.5) : len(timeSeries)]
    return Train, Test


def log_returns(dataframe):
    """
    @params:
        dataframe: pandas dataframe of our stock.
    @returns:
        diff: logaritmic difference. 
    """
    diff = np.log(dataframe).diff().dropna()
    return diff


def MC(start_val, dataframe, nofsim):
    """
    @params:
        dataframe: pandas dataframe of our stock
    @returns:
        
    """
    #Parameters
    mu = dataframe.mean()
    var = dataframe.var()
    drift = mu - 0.5*var
    std = dataframe.std()
    days = np.arange(252)

    #Variables
    epsilon = norm.ppf(np.random.rand(len(days), nofsim))
    delta_x = drift.values + std.values*epsilon 
    sim_values = np.zeros_like(delta_x)
    sim_values[0] = start_val
    
    #Simulation
    for t in range(1, len(days)):
        sim_values[t] = sim_values[t-1]*np.exp(delta_x[t])
    return sim_values
    



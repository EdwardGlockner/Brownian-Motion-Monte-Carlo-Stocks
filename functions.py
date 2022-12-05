import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm 


def read_csv(file_path, tick):
    """
    @params:
        file_path: path to the destination of the file
        tick: ticker of the stock

    @returns:
        temp: pandas dataframe with closing price and date of the stock

    Reads a file_path for a given ticker, and returns a dataframe with a column: close.
    """
    temp = pd.read_csv(file_path, parse_dates=[0], index_col=0)
    temp.rename(columns={"Close" : tick})
    temp.drop(["Open", "High", "Low", "Adj Close", "Volume"], axis = 1, inplace = True)
    return temp

def log_returns(dataframe):
    """
    @params:
        dataframe: pandas dataframe of our stock.
    @returns:
        diff: logaritmic difference. 
    """
    diff = np.log(dataframe).diff().dropna()
    return diff


def MC(dataframe,nofsim):
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
    delta_x = drift.values + var.values*epsilon 
    sim_values = np.zeros_like(delta_x)
    value_0 = dataframe["Close"].iloc[0]
    sim_values[0] = value_0
    
    #Simulation
    for t in range(1, len(days)):
        sim_values[t] = sim_values[t-1]*np.exp(delta_x[t])
    
    print(sim_values)
    





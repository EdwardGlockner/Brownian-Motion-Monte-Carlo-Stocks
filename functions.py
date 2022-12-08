from scipy.stats import norm
from scipy import stats as st
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


def MC(train, start_val, dataframe, nofsim, days_sim):
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
    days = np.arange(days_sim)

    #Variables
    epsilon = norm.ppf(np.random.rand(len(days), nofsim))
    delta_x = drift.values + std.values*epsilon 
    sim_values = np.zeros_like(delta_x)
    sim_values[0] = start_val
    
    #Simulation
    for t in range(1, len(days)):
        sim_values[t] = sim_values[t-1]*np.exp(delta_x[t])
    
    index_values = [i for i in range(0, len(days))]
    column_index = [i for i in range(0, nofsim)]
    new_dataframe = pd.DataFrame(sim_values)

    dates_train = pd.to_datetime(train.index.values)
    dates_sim = pd.date_range(str(dates_train[-1]), periods = days_sim, freq="D")
    new_dataframe.index = dates_sim

    return new_dataframe
    

def plot_tree(train, days_sim, mc_sim):
    dates_train = pd.to_datetime(train.index.values)
    dates_sim = pd.to_datetime(mc_sim.index.values)

    
    train.loc[dates_train[0] : dates_train[-1], "Close"].plot()

    for i in range(0, len(mc_sim.columns)):
        mc_sim.loc[dates_sim[0] : dates_sim[-1], i].plot(lw=0.3)

    plt.show()

def plot_dataframe(df, xlabel, ylabel, title, file_output):
    fig = df.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if file_output != "":
        fig = fig.get_figure()
        fig.savefig(file_output)
    plt.show()
    

def final_values(mc_sim):
    last_values = mc_sim.iloc[-1,:]
    return last_values

def plot_histogram(values):
    values.hist(bins=40, grid=True, figsize=(7,4), color = "#86bf91", zorder=2, rwidth=0.9)
    plt.xlabel("Stock value")
    plt.ylabel("Frequency")
    plt.title("title")
    plt.show()

def conf_interval(start_val, dataframe, nofsim):
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
    delta_x_interval_1 = np.zeros(len(days))
    delta_x_interval_2 = np.zeros(len(days))
    expected_delta_x = np.zeros(len(days))

    #Defining confidence intervals
    for t in range(1, len(days)):
        delta_x_interval_1[t] = drift*(t-days[0]) + std *1.96*np.sqrt(t - days[0])
        delta_x_interval_2[t] = drift*(t-days[0]) + std *-1.96*np.sqrt(t - days[0])

    S = np.zeros_like(delta_x)
    S_interval_1 = np.zeros_like(delta_x_interval_1)
    S_interval_2 = np.zeros_like(delta_x_interval_2)
    S_interval_1[0] = start_val
    S_interval_2[0] = start_val
    S[0] = start_val
    expected_delta_x[0] = start_val

    #Simulation of confidence interval
    for t in range(1, len(days)):
        S[t] = S[t-1]*np.exp(delta_x[t])
        S_interval_1[t] = start_val*np.exp(delta_x_interval_1[t])
        S_interval_2[t] = start_val*np.exp(delta_x_interval_2[t])
        expected_delta_x[t] = expected_delta_x[t-1]*np.exp(mu.values)
    
    return S,S_interval_1,S_interval_2,expected_delta_x


def plot_conf(S,S_interval_1,S_interval_2,expected_delta_x):
    plt.figure(figsize=(12.2,4.5))
    color = 'black'
    plt.plot(S)
    plt.plot(S_interval_1,color=color)
    plt.plot(S_interval_2,color=color)
    plt.plot(expected_delta_x,color=color)
    plt.title('Price of Amazon stock 1 year from now: 95% Confidence Interval')
    plt.xlabel('Time',fontsize=18)
    plt.ylabel('Price',fontsize=18)
    plt.show()


def conf_interval_values(last_values):
    last_values = last_values.to_numpy()
    interval = st.t.interval(alpha = 0.99, df = len(last_values)-1, loc = last_values.mean(), scale = st.sem(last_values))
    return interval

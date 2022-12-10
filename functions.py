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
    Splits a dataframe into a training set and a testing set, with 50 % of the data in each
    @params:
        timeSeris: the dataframe that will be split

    @returns:
        Train:the training set
        Test: the testing set

    Splits a dataframe into a training set containing 80% of the data, and a testing set containing 
    """
    Train = timeSeries[0 : round(len(timeSeries)*0.5)]
    Test = timeSeries[round(len(timeSeries)*0.5) : len(timeSeries)]
    return Train, Test


def log_returns(dataframe):
    """
    Calculates the daily logarithmic returns
    @params:
        dataframe: pandas dataframe of our stock.
    @returns:
        diff: logaritmic difference. 
    """
    diff = np.log(dataframe).diff().dropna()
    return diff


def MC(train, start_val, dataframe, nofsim, days_sim):
    """
    Monte Carlo simulate the stock through the Brownian Motion model
    @params:
        train: the training set
        start_val: starting value of the simulation (the last value in the training set)
        dataframe: pandas dataframe of our stocks logarthmic daily returns
        nofsim: number of Monte-Carlo simulations
        days_sim: number of days forward that will be simulated

    @returns:
        new_dataframe: All the Monte-Carlo simulations
       

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
    """
    Plots the training set together with all the Monte-Carlo simulations
    @params:
        train: the training set
        days_sim: number of days simulated by the Monte-Carlo simulations
        mc_sim: all the simulated paths

    @returns:
        none
    """
    dates_train = pd.to_datetime(train.index.values)
    dates_sim = pd.to_datetime(mc_sim.index.values)

    
    train.loc[dates_train[0] : dates_train[-1], "Close"].plot()

    for i in range(0, len(mc_sim.columns)):
        mc_sim.loc[dates_sim[0] : dates_sim[-1], i].plot(lw=1)
    plt.ylabel("Stock price")
    plt.title("Monte Carlo simulation")
    plt.show()

def plot_dataframe(df, xlabel, ylabel, title, file_output):
    """
    Plots a dataframe
    @params:
        df: dataframe that should be plotted
        xlabel: x axis label of the plot
        ylabel: y axis label of the plot
        title: title of the plot
        file_output: path to a file that should be saved (can be set to null)

    @returns:
        none
    """
    fig = df.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if file_output != "":
        fig = fig.get_figure()
        fig.savefig(file_output)
    plt.show()
    

def final_values(mc_sim):
    """
    Gets all the final values of the Monte-Carlo simulated price paths
    @params:
        mc_sim: the Monte-Carlo simulated paths in a pandas dataframe

    @returns:
        last_values: all the final values
    """
    last_values = mc_sim.iloc[-1,:]
    return last_values

def plot_histogram(values):
    """
    Plots a histogram of all the final values of the Monte-Carlo simualted price paths
    @params:
        values: the final values of the pandas dataframe

    @returns:
        none
    """
    values.hist(bins=40, grid=True, figsize=(7,4), color = "#86bf91", zorder=2, rwidth=0.9)
    plt.xlabel("Stock value")
    plt.ylabel("Frequency")
    plt.title("Histogram of the final stock values of the Monte-Carlo simulation")
    plt.show()

def conf_interval(start_val, dataframe, nofsim):
    """
    Monte Carlo simulation with confidence interval of the drift in the Brownian Motion model
    @params:
        start_val: starting value of the simulation (last value in the training set)
        dataframe: pandas dataframe of our stock
        nofsim: number of Monte-Carlo simulated price paths

    @returns:
        S: the simulated price paths
        S_interval_1: the higher confidence interval
        S_interval_2: the lower confidence interval
        expected_delta_x: the excpected logarthmic daily returns
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
    """
    Plots the simulated price paths together with the confidence interval
    @params:
        S: the simulated price paths of the Monte-Carlo simulation
        S_interval_1: the higher confidence interval
        S_interval_2: the lower confidence interval
        excpected_delta_x: the excpected logarthmic daily returns

    @returns:
        none
    """
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
    """
    Creates a confidence interval of the final values of the Monte-Carlo simulated price paths
    @params:
        last_values; the final values of all the Monte-Carlo simulated price paths

    @returns:
        interval: the 99% confidence interval
        
    """
    last_values = last_values.to_numpy()
    interval = st.t.interval(alpha = 0.99, df = len(last_values)-1, loc = last_values.mean(), scale = st.sem(last_values))
    return interval

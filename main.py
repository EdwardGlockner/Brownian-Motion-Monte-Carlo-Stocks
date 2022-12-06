from functions import *
import scipy
from matplotlib import pyplot as plt
# Data of the stock
ticker = "AMZN"
file_path = "Data/AMZN.csv" 
#file_path = "Brownian-Motion-Monte-Carlo-Stocks\Data\AMZN.csv"

# Reads the data into a dataframe
data = read_csv(file_path, ticker)
train, test = split_timeseries(data)

train_returns = log_returns(train)
start_val = train["Close"].iloc[len(train)-1]
mc_sim = MC(start_val, train_returns, 10000)

plt.figure()
plt.plot(mc_sim)
plt.show()




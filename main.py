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
mc_sim = MC(train_returns, 10000)
mc_price = convert_to_price(train_returns["Close"].iloc[-1], mc_sim)

plt.figure()
plt.plot(mc_price)
plt.show()

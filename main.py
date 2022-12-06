from functions import *

import scipy
from matplotlib import pyplot as plt
# Data of the stock
ticker = "AMZN"
file_path = "/Users/edwardglockner/Library/CloudStorage/OneDrive-Uppsalauniversitet/Fristående Kurser/Inferensteori I/Brownian-Motion-Monte-Carlo-Stocks/Data/AMZN.csv" 


# Reads the data into a dataframe
data = read_csv(file_path, ticker)
train, test = split_timeseries(data)

train_returns = log_returns(train)
mc_sim = MC_V2(train_returns, test, 1)

plt.figure()
plt.plot(mc_sim)
plt.show()

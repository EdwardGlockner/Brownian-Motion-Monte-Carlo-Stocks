from functions import *
import scipy

import datetime
# Data of the stock
ticker = "AMZN"
file_path = "Data/AMZN.csv"
#file_path = "Brownian-Motion-Monte-Carlo-Stocks\Data\AMZN.csv"

# Reads the data into a dataframe
data = read_csv(file_path, ticker)
train, test = split_timeseries(data)
train_returns = log_returns(train)
start_val = train["Close"].iloc[len(train)-1]
mc_sim = MC(start_val, train_returns, 2)

dates_train = pd.to_datetime(train.index.values)
dates_sim = pd.date_range(str(dates_train[-1]), periods=len(mc_sim), freq='D')

mc_sim.index = dates_sim

merged = pd.concat([train, mc_sim])

plt.figure()
train.loc[dates_train[0] : dates_train[-1], "Close"].plot()
mc_sim.loc[dates_sim[0] : dates_sim[-1], 0].plot()
mc_sim.loc[dates_sim[0] : dates_sim[-1], 1].plot()
plt.show()

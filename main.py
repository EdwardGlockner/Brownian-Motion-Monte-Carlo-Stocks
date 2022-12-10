from functions import *
import scipy

import datetime
# Data of the stock
ticker = "AMZN"
file_path = "Data/AMZN.csv"
#file_path = "Brownian-Motion-Monte-Carlo-Stocks\Data\AMZN.csv" # for windows

# Reads the data into a dataframe
data = read_csv(file_path, ticker)
plot_dataframe(data, "Date", "Closing price", "Historical data of Amazon stock", "Images/historical_data.png")

train, test = split_timeseries(data)
train_returns = log_returns(train)
start_val = train["Close"].iloc[len(train)-1]

days_sim = 121
mc_sim = MC(train, start_val, train_returns, nofsim=1000, days_sim=days_sim)

plot_tree(train, days_sim, mc_sim)
S, S_interval_1, S_interval_2, expected_delta_x = conf_interval(start_val, train, 5000)
plot_conf(S, S_interval_1, S_interval_2, expected_delta_x)
final_values = final_values(mc_sim)
mean_final_value = final_values.mean()
final_date = mc_sim.last_valid_index()
correct_final_value = test.iloc[days_sim]["Close"]
plot_histogram(final_values)

print("Mean simulated value: ", mean_final_value)
print("Correct final value: ", correct_final_value)

interval = conf_interval_values(final_values)

print("Interval", interval)
plot_histogram(final_values)

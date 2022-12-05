from functions import *


# Data of the stock
ticker = "AMZN"
file_path = "Brownian-Motion-Monte-Carlo-Stocks\Data\AMZN.csv"


# Reads the data into a dataframe
data = read_csv(file_path, ticker)

diff = log_returns(data)

sim = MC(diff,2)




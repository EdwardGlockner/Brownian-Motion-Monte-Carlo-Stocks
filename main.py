from functions import *

ticker = "AMZN"
file_path = "Brownian-Motion-Monte-Carlo-Stocks\Data\AMZN.csv"

data = read_csv(file_path, ticker)

diff = log_returns(data)
print(diff)




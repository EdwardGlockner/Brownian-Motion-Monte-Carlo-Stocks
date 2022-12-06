from functions import *

import scipy
from matplotlib import pyplot as plt
# Data of the stock
ticker = "AMZN"
file_path = "/Users/edwardglockner/Library/CloudStorage/OneDrive-Uppsalauniversitet/Fristående Kurser/Inferensteori I/Brownian-Motion-Monte-Carlo-Stocks/Data/AMZN.csv" 


# Reads the data into a dataframe
data = read_csv(file_path, ticker)
train, test = split_timeseries(data)

<<<<<<< HEAD
diff = log_returns(data)

sim = MC(diff,1)


=======
train_returns = log_returns(train)
mc_sim = MC(train_returns, 1)
>>>>>>> 45de7057ed3b7840c8d44488562141a5ab9ff5d4

plt.figure()
plt.plot(mc_sim)
plt.show()

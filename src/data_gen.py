import pandas as pd
import numpy as np

# create a dummy array
arr = np.round(np.random.rand(100, 2), decimals=2)

# convert array into dataframe
df = pd.DataFrame(arr, columns=["Value", "Weight"])

# save the dataframe as a csv file
df.to_csv("data/knapsack.csv")
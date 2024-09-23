# =============================================================================
# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt    
# =============================================================================
# df = pd.read_csv("S1_Session1Data.data") 
cols = ["transaction date", "house age", "distance to the nearest MRT station", "number of convenience stores", "latitude", "longitude", "house price of unit area"]        # name of the columns from the other file called "Session1Data.names"
df = pd.read_csv("S4_Session4Data.csv", names=cols)      # reading the dataset again, but this time we are naming the columns
df.head()       
# =============================================================================
# for i in range(len(df.columns)):
for label in df.columns[0:-1]:
    plt.scatter(df[label], df["house price of unit area"])
    plt.title(label)
    plt.show()
# in the third plot, we are plotting "hous price" (continuous variable) vs 
# "number of stores" (discrete variable). That is why we see this weird plot. 
# Since we are focusing on regression now, we can remove this column
# Also, the first column doesn't look discrete at the first glance. However, 
# it just includes some specific numbers that we can round them to integers.
df=df.drop(["number of convenience stores"], axis='columns')
df=df.drop(["transaction date"], axis='columns')
# now, if we plot again, we will have:
for label in df.columns[0:-1]:
    plt.scatter(df[label], df["house price of unit area"])
    plt.title(label)
    plt.show()
# =============================================================================
train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

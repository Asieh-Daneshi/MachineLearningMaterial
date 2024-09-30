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

df.columns=cols 
df.head()      
# =============================================================================
# here, we plot the price of the houses based on different features
# for i in range(len(df.columns)):
for label in df.columns[0:-1]:
    plt.scatter(df[label], df["house price of unit area"])
    plt.title(label)
    plt.show()
# in the third plot, we are plotting "hous price" (continuous variable) vs 
# "number of stores" (discrete variable). That is why we see this weird plot. 
# Since we are focusing on regression now, we can remove this column
# Also, the first column doesn't look discrete at the first glance. However, 
# it represents transaction dates, that is alsp discrete.
df=df.drop(["number of convenience stores"], axis='columns')
df=df.drop(["transaction date"], axis='columns')
# now, if we plot again, we will have:
for label in df.columns[0:-1]:
    plt.scatter(df[label], df["house price of unit area"])
    plt.title(label)
    plt.show()
# =============================================================================
train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

# Now, imagine that every time we are working on a specific feature, we want
# to have just that feature and the output, and nothing else. So, let's write a
# function to do so
import copy
def myXY(dataframe, yLabel, xLabel=None):
    # here we keep a backup of our dataframe. what is the difference between 
    # deepcopy and copy?
    # try the code at the end of this code
    dataframe=copy.deepcopy(dataframe)

    if len(xLabel)==1:
        X=dataframe[xLabel].values.reshape(-1,1)
    else:
        X=dataframe[xLabel].values
    Y=dataframe[yLabel].values.reshape(-1,1)
    data=np.hstack((X,Y))
    
    return data, X, Y

_,xTrain, yTrain=myXY(train, ["house price of unit area"],["latitude"])


# the difference between deepcopy and copy ====================================
x1=[1, 2, [3, 4], 5]
x2=copy.deepcopy(x1)
x3=copy.copy(x1)
x2[2][0]=6
print(x2)
print(x1)
x3[2][0]=6
print(x3)
print(x1)
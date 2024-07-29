# =============================================================================
# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt    
# =============================================================================
df = pd.read_csv("S1_Session1Data.data") 

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]        # name of the columns from the other file called "Session1Data.names"
df = pd.read_csv("S1_Session1Data.data", names=cols)      # reading the dataset again, but this time we are naming the columns
df.head()       
df["class"].unique()

# =============================================================================
# Changing the character labels into numbers
uniqueClasses=df["class"].unique()        
df["class"] = (df["class"] == "g").astype(int)      
# =============================================================================
for label in cols[:-1]:         
    print(label)
    plt.hist(df[df["class"]==1][label], color='blue', label='gamma', alpha=0.7, density=True)   
    plt.hist(df[df["class"]==0][label], color='green', label='hadron', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()
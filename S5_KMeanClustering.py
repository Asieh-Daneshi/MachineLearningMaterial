import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


cols=["area","perimeter","compactness","length","width","asymmetry","groove","class"]
df=pd.read_csv('seeds_dataset.txt', delimiter='\t')
import seaborn as sns

import seaborn as sns
for i in range(len(cols)-1):
    for j in range(i+1,len(cols)-1):
        xLabel=cols[i]
        yLabel=cols[j]
        sns.scatterplot(x=xLabel, y=yLabel, data=df, hue='class')
        plt.show()
        
# simple k-mean Clustering
from sklearn.cluster import KMeans
x="area"
y="perimeter"
X=df[[x,y]].values
kmeans=KMeans(n_clusters=3).fit(X)
clusters=kmeans.labels_
cluster_df=pd.DataFrame(np.hstack((X,clusters.reshape(-1,1))),columns=[x,y,"class"])
sns.scatterplot(x=x, y=y, data=cluster_df, hue='class')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
sns.scatterplot(x=x, y=y, ax=ax1, data=df, hue='class')
sns.scatterplot(x=x, y=y, ax=ax2, data=cluster_df, hue='class')

# multi-dimension k-mean Clustering
X=df[cols[:-1]].values
kmeans=KMeans(n_clusters=3).fit(X)
clusters=kmeans.labels_
cluster_df=pd.DataFrame(np.hstack((X,clusters.reshape(-1,1))),columns=df.columns)
sns.scatterplot(x=x, y=y, data=cluster_df, hue='class')

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# sns.scatterplot(x=x, y=y, ax=ax1, data=df, hue='class')
# sns.scatterplot(x=x, y=y, ax=ax2, data=cluster_df, hue='class')
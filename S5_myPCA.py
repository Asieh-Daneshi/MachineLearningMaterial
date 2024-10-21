import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


cols=["area","perimeter","compactness","length","width","asymmetry","groove","class"]
df=pd.read_csv('seeds_dataset.txt', delimiter='\t')
import seaborn as sns

import seaborn as sns
        
# simple k-mean Clustering
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
PCAfitted_x=pca.fit_transform(X)
plt.scatter(PCAfitted_x[:,0], PCAfitted_x[:,1])
plt.show()



from sklearn.cluster import KMeans
x="area"
y="perimeter"
X=df[[x,y]].values
kmeans=KMeans(n_clusters=3).fit(X)
clusters=kmeans.labels_
cluster_df=pd.DataFrame(np.hstack((X,clusters.reshape(-1,1))),columns=[x,y,"class"])
sns.scatterplot(x=x, y=y, data=cluster_df, hue='class')

kmeans_PCA_df=pd.DataFrame(np.hstack((PCAfitted_x, kmeans.labels_.reshape(-1,1))), columns=["pca1","pca2","class"])
kmeans_original_df=pd.DataFrame(np.hstack((PCAfitted_x, df["class"].values.reshape(-1,1))), columns=["pca1","pca2","class"])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
sns.scatterplot(x="pca1", y="pca2", ax=ax1, data=kmeans_PCA_df, hue='class')
sns.scatterplot(x="pca1", y="pca2", ax=ax2, data=kmeans_original_df, hue='class')
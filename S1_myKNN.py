# Importing required libraries
import numpy as np
import pandas as pd
# conda install pandas
import matplotlib.pyplot as plt 
# conda install matplotlib
from sklearn.preprocessing import StandardScaler
# conda install scikit-learn   # for sklearn
# conda install preprocessing
from imblearn.over_sampling import RandomOverSampler
# conda install imbalanced-learn
from imblearn.under_sampling import RandomUnderSampler
# =============================================================================
# Changing the character labels into numbers
uniqueClasses=df["class"].unique()     
# First way (one-hot encoding)
# df["class"] = (df["class"] == "g").astype(int)      # compares the elements in the "class" column with "g" and returns "True" or "Flase". Then converts these binaries into integers 1 and 0
# Second way
# df["class"] = df['class'].astype('category').cat.codes
# Third way (comprehensive)
labels = df["class"]
label_mapping = {uniqueClasses[0]:0, uniqueClasses[1]:1}
numerical_labels = [label_mapping[label] for label in labels]
df["class"]=numerical_labels
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
# =============================================================================
# Train, validation, test datasets
# First 60% as train, from 60% to 80% as validation, the rest as test
train, valid, test = np.split(df.sample(frac=1, replace=False), [int(0.6*len(df)), int(0.8*len(df))])
# =============================================================================
# Writing a function that gets data, z-scores each column of data, and balances them (either oversample or undersample), in a way that all of them have the same number of elements
def scale_dataset(dataframe, oversample, undersample):
    X = dataframe[dataframe.columns[:-1]].values    
    Y = dataframe[dataframe.columns[-1]].values     
    
    scaler = StandardScaler()       
    X = scaler.fit_transform(X)     
    if oversample:
        ros = RandomOverSampler()   
        X, Y = ros.fit_resample(X, Y)       
        
    if undersample:
        rus = RandomUnderSampler()   
        X, Y = rus.fit_resample(X, Y)       
        
    data = np.hstack((X, np.reshape(Y, (-1,1))))        
    return data, X, Y
# =============================================================================
# Testing the function that we wrote in the previous block
# =============================================================================
train, Xtrain, Ytrain = scale_dataset(train, oversample=True, undersample=False)
valid, Xvalid, Yvalid = scale_dataset(valid, oversample=False, undersample=False)
test, Xtest, Ytest = scale_dataset(test, oversample=False, undersample=False)
# *****************************************************************************
# kNN
# *****************************************************************************
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(Xtrain, Ytrain)
# KNeighborsClassifier()
Ypred = knn_model.predict(Xtest)
# print(Ypred)
# print(Ytest)
from sklearn.metrics import classification_report
print(classification_report(Ytest,Ypred))


# Evaluating the accuracy of our classification (just accuracy, not the whole report!)

from sklearn.metrics import accuracy_score
accuracy_score(Ytest, Ypred)

# Testing different number of neighbors for KNN to find the best one!

from sklearn.model_selection import cross_val_score

Ks = np.arange(3,20,2)
best_score = -1
best_K = None

for k in Ks:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, Xtrain, Ytrain, cv=5)  # 5-fold cross-validation
    average_score = np.mean(scores)

    if average_score > best_score:
        best_score = average_score
        best_K = k

print("Best score is: ", best_score)
print("Best number of neighbors: ", best_K)

# Testing different algorithms for finding the distance between points on KNN to find the best one!

from sklearn.model_selection import cross_val_score


best_score = -1
best_K = None

for k in Ks:
    knn_model = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(knn_model, Xtrain, Ytrain, cv=5)  # 5-fold cross-validation
    average_score = np.mean(scores)

    if average_score > best_score:
        best_score = average_score
        best_K = k

print("Best score is: ", best_score)
print("Best number of neighbors: ", best_K)

knn_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
scores = cross_val_score(knn_model, Xtrain, Ytrain, cv=5)  # 5-fold cross-validation
average_score = np.mean(scores)

if average_score > best_score:
    best_score = average_score
    best_distance = 'euclidean'
# ----------------------------------------------------------------------------------
knn_model = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
scores = cross_val_score(knn_model, Xtrain, Ytrain, cv=5)  # 5-fold cross-validation
average_score = np.mean(scores)

if average_score > best_score:
    best_score = average_score
    best_distance = 'manhattan'
# ----------------------------------------------------------------------------------
knn_model = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
scores = cross_val_score(knn_model, Xtrain, Ytrain, cv=5)  # 5-fold cross-validation
average_score = np.mean(scores)

if average_score > best_score:
    best_score = average_score
    best_distance = 'minkowski'
# ----------------------------------------------------------------------------------
knn_model = KNeighborsClassifier(n_neighbors=3, metric='chebyshev')
scores = cross_val_score(knn_model, Xtrain, Ytrain, cv=5)  # 5-fold cross-validation
average_score = np.mean(scores)

if average_score > best_score:
    best_score = average_score
    best_distance = 'chebyshev'
# ----------------------------------------------------------------------------------
# knn_model = KNeighborsClassifier(n_neighbors=11, metric='mahalanobis')
# knn_model.fit(Xtrain, Ytrain)
# Ypred = knn_model.predict(Xtest)
# allAccuracies.append(accuracy_score(Ytest, Ypred))
# ----------------------------------------------------------------------------------
knn_model = KNeighborsClassifier(n_neighbors=3, metric='hamming')
scores = cross_val_score(knn_model, Xtrain, Ytrain, cv=5)  # 5-fold cross-validation
average_score = np.mean(scores)

if average_score > best_score:
    best_score = average_score
    best_distance = 'hamming'
# ----------------------------------------------------------------------------------
knn_model = KNeighborsClassifier(n_neighbors=3, metric='canberra')
scores = cross_val_score(knn_model, Xtrain, Ytrain, cv=5)  # 5-fold cross-validation
average_score = np.mean(scores)

if average_score > best_score:
    best_score = average_score
    best_distance = 'canberra'
# ----------------------------------------------------------------------------------
knn_model = KNeighborsClassifier(n_neighbors=3, metric='cosine')
scores = cross_val_score(knn_model, Xtrain, Ytrain, cv=5)  # 5-fold cross-validation
average_score = np.mean(scores)

if average_score > best_score:
    best_score = average_score
    best_distance = 'cosine'
# ----------------------------------------------------------------------------------
knn_model = KNeighborsClassifier(n_neighbors=3, metric='jaccard')
scores = cross_val_score(knn_model, Xtrain, Ytrain, cv=5)  # 5-fold cross-validation
average_score = np.mean(scores)

if average_score > best_score:
    best_score = average_score
    best_distance = 'jaccard'
# ----------------------------------------------------------------------------------
knn_model = KNeighborsClassifier(n_neighbors=3, metric='braycurtis')
scores = cross_val_score(knn_model, Xtrain, Ytrain, cv=5)  # 5-fold cross-validation
average_score = np.mean(scores)

if average_score > best_score:
    best_score = average_score
    best_distance = 'braycurtis'
# ----------------------------------------------------------------------------------
# knn_model = KNeighborsClassifier(n_neighbors=11, metric='haversine')
# knn_model.fit(Xtrain, Ytrain)
# Ypred = knn_model.predict(Xtest)
# allAccuracies.append(accuracy_score(Ytest, Ypred))


print(best_distance)
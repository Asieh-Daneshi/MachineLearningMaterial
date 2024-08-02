# *****************************************************************************
# Naive Bayes
# *****************************************************************************
from sklearn.naive_bayes import GaussianNB

nb_model=GaussianNB()        # here, we introduce nb_model, which is the GaussianNB
nb_model.fit(Xtrain, Ytrain)
Ypred = nb_model.predict(Xtest)

from sklearn.metrics import classification_report
print(classification_report(Ytest,Ypred))
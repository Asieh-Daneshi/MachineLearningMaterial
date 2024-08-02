# *****************************************************************************
# SVM
# *****************************************************************************
from sklearn.linear_model import LogisticRegression

lg_model=LogisticRegression()        # here, we introduce nb_model, which is the GaussianNB
lg_model.fit(Xtrain, Ytrain)
Ypred = lg_model.predict(Xtest)

from sklearn.metrics import classification_report
print(classification_report(Ytest,Ypred))
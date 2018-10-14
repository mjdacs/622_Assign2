
import numpy as np
import pandas as pd 

from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn import metrics

from train_model import clean_and_train



def main():
    """The main function of this file, which scores the random forest model.
    
    Parameters
    ----------
    No parameters are taken. The function pulls in the dataset from 
    train_model.py, loads the RandomForestClassifer via the joblib libray,
    predicts and scores the model.

    
    Returns
    -------
    Nothing is returned. Accuracy of the model is printed to the console.
    """

    X_train, X_test, y_train, y_test = clean_and_train()

    clf = joblib.load('model.joblib') 

    model = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))




if __name__ == '__main__':
    main()
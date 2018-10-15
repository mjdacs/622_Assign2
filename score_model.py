
import numpy as np
import pandas as pd 

from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn import metrics

from train_model import clean_and_train
import time


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

    clf.fit(X_train, y_train)
    print("Fitting model...")
    time.sleep(3)

    y_pred = clf.predict(X_test)
    
    print("Model fit and predictions made.\n")
    print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred), '\n')
    print(classification_report(y_test, y_pred, target_names=['Perished', 'Survived']))
    
    # Write results and output reports to csv and textfile, respectively
    results = pd.DataFrame({'predicted': y_pred,
                            'actual': y_test})
    results.to_csv('results.csv')

    
    print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred), '\n', 
        file=open("output.txt", "a"))     
    print(classification_report(y_test, y_pred, target_names=['Perished', 'Survived']),
        file=open("output.txt", "a"))   
    print("The model's outputs were written to output.txt and it's results to results.txt.")
   



if __name__ == '__main__':
    main()
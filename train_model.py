# standard data science libraries
import numpy as np
import pandas as pd 

# Model algorithm
from sklearn.ensemble import RandomForestClassifier
#
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline


# pull and check data
import pull_data as pull


def main():
    
    # TURN ON LATER
    train_set, test_set = pull.pull_data()

    # train_set = pd.read_csv("train.csv")
    # test_set = pd.read_csv("test.csv")

    
    full_data = pd.concat([train_set, test_set], sort=True)
    
    data_cleaner = [train_set, test_set]


    
    train_set, test_set = impute_and_drop(train_set, test_set)




# I have chosen to impute the median in the Age column, Mode in Embarked, and the median in Fare
# Age had a fairly large left skew so I thought it best to use the median CHECK
# For Embarked, there were only three ports the titanic left from, only two missing values, and most 
# were out of one port.
# The median was chosen also to impute over Fare.
# Lastly Cabin, PassengerId, and Ticket were dropped, since they dont appear to provide much value    
def impute_and_drop(train_set, test_set):
    
    print("")
    print("Checking for null values: train & test")
    print("=" * 20)
    print(train_set.isnull().sum())
    print("-" * 10)
    print(test_set.isnull().sum())
    print("=" * 20)
    
    for dataset in [train_set, test_set]:
        
        dataset['Age']      = lambda x: x['Age'].fillna(x['Age'].median(), inplace = True)
        dataset['Embarked'] = lambda x: x['Embarked'].fillna(x['Embarked'].mode(), inplace = True)
        dataset['Fare']     = lambda x: x['Fare'].fillna(x['Fare'].median(), inplace = True)
        
    train_set.drop(['Cabin', 'PassengerId', 'Ticket'], axis=1, inplace=True) # drop from training set only
    
    print("")
    print("Age, Embarked, and Fare successfully imputed in the training & test sets.\n",
    "Cabin, PassengerId, and Ticket were successfully dropped in the training set.")
    print("=" * 20)
    print(train_set.isnull().sum())
    print("-" * 10)
    print(test_set.isnull().sum())
    print("=" * 20)
    
    return train_set, test_set

    # imp = Imputer(missing_values='Nan', strategy='mean', axis=0)

    # clf = SVC()

    # steps = [('imputation', imp),
    #         ('SVM', clf)]

    # pipeline = Pipeline(steps)

    # X_train, X_test, y_train, y_test = train_test_split()

    # pipeline.fit(X_train,  y_train)

    











if __name__ == "__main__":
    main()
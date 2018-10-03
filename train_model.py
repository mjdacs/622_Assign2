# standard data science libraries
import numpy as np
import pandas as pd 

# Model algorithm
import sklearn
from sklearn.ensemble import RandomForestClassifier
#
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.pipeline import Pipeline


# pull and check data
# import pull_data as pull


def main():
    
    # TURN ON LATER
    # train_set, test_set = pull.pull_data()

    train_set = pd.read_csv("train.csv")
    test_set = pd.read_csv("test.csv")

    
    full_data = pd.concat([train_set, test_set], sort=True)
    
    data_cleaner = [train_set, test_set]

    print("")
    print("Checking for null values: train & test")
    print("=" * 20)
    print(train_set.isnull().sum())
    print("-" * 10)
    print(test_set.isnull().sum())
    print("=" * 20)
    
    train_set, test_set = impute_and_drop(train_set, test_set)




 
def impute_and_drop(train_set, test_set):
    """Impute and drop relevant variables
    
    Parameters
    ----------
    train_set : DataFrame of the training set
    test_set : DataFrame of the test set
        Impute the median for NaNs in the Age column, mode in the Embarked
        column, and median in the Fare column.
        Drop Cabin, PassengerId, and Ticket columns in the training set.
        
    Returns
    -------
    train_set : Imputed DataFrame of the training set
    test_set : Imputed DataFrame of the test set
    """  
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



def feature_engineer(train_set, test_set):

    for dataset in [train_set, test_set]:

        pass


def create_dummy_variables(train_set, test_set):
    label = LabelEncoder
    label.fit_transform
    for dataset in [train_set, test_set]:

        pass

























    # imp = Imputer(missing_values='Nan', strategy='mean', axis=0)

    # clf = SVC()

    # steps = [('imputation', imp),
    #         ('SVM', clf)]

    # pipeline = Pipeline(steps)

    # X_train, X_test, y_train, y_test = train_test_split()

    # pipeline.fit(X_train,  y_train)







if __name__ == "__main__":
    main()
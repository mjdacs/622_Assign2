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

    # peak at null values before imputation
    print('\n',
    "Checking for null values: train & test.\n", 
    "=" *20, '\n',
    train_set.isnull().sum(), '\n',
    "=" *10, '\n',
    test_set.isnull().sum(), '\n',
    "=" *20)
    
    # impute and drop training and test sets
    train_set, test_set = impute_and_drop(train_set, test_set)

    # confirm imputation and dropped columns
    print('\n',
    "Age, Embarked, and Fare successfully imputed in the training & test sets.\n",
    "Cabin, PassengerId, and Ticket were successfully dropped in the training set.\n", 
    "=" *20, '\n',
    train_set.isnull().sum(), '\n',
    "=" *10, '\n',
    test_set.isnull().sum(), '\n',
    "=" *20)

    print(train_set.info(),
    test_set.info(),
    train_set.sample(10))

    print(test_set.Age.head())

 
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
        
        dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
        dataset['Embarked'].fillna(dataset['Embarked'].mode(), inplace = True)
        dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
        
    train_set.drop(['Cabin', 'PassengerId', 'Ticket'], axis=1, inplace=True) # drop from training set only
     
    return train_set, test_set



def feature_engineer(train_set, test_set):

    for dataset in [train_set, test_set]:

        dataset['FamilySize'] = lambda x: x['SibSp']


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
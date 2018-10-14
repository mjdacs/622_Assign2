# standard data science libraries
import numpy as np
import pandas as pd 

# Model algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from sklearn.preprocessing import LabelEncoder


# pull and check data
import pull_data as pull


def clean_and_train():
    """The main function of this file. Cleans the data and trains the model
    
    Parameters
    ----------
    No parameters are taken. The function is meant to run the rest of the
    functions in the program which are designed to impute the dataset,
    engineer new features, and encode some of the variables.

    After the dataset is cleaned, the training set is itself split into a
    training and test set, due to there not existing a target variable in
    the test csv. It was considered to submit scoring to kaggle to obtain
    accuracy results but it was not part of the requirements. The only way
    to obtain accuracy is to split the local training set 80/20 and score
    against the test portion. You can see this being the focus starting
    with line 56 were the train set is isolated from full_data.    

    The model is fitted and insead of .pkl, sklearn's joblib function was
    used to save the model locally.
        
    Returns
    -------
    X_train, X_test, y_train, y_test: All four parameters of the data set
    were returned for score_model.py to pull and predict on.
    """  
    
    # TURN ON LATER
    train, test = pull.pull_data()

    # train = pd.read_csv("train.csv")
    # test = pd.read_csv("test.csv")
    full_data = pd.concat([train, test], keys=["train", "test"],sort=True)
    
    titanic = (full_data
     .pipe(impute)
     .pipe(feature_eng)
     .pipe(encode)
     .drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=False))
  
    df = (titanic.loc['train']
      .drop(['Survived'], axis=1))
    
    y = titanic.loc['train']['Survived']

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

    clf = RandomForestClassifier(n_estimators=100)

    joblib.dump(clf, 'model.joblib')

    return X_train, X_test, y_train, y_test
    


def impute(df):
    """Impute and drop relevant variables
    
    Parameters
    ----------
    df : DataFrame of the training or test set
        Impute the median for NaNs in the Age column, mode in the Embarked
        column, and median in the Fare column.
        Drop Cabin, PassengerId, and Ticket columns in the training set.
        
    Returns
    -------
    df: Imputed DataFrame of the training or test set
    """  
       
    df['Age'].fillna(df['Age'].median(), inplace = True)
    df['Embarked'].fillna(df['Embarked'].mode(), inplace = True)
    df['Fare'].fillna(df['Fare'].median(), inplace = True)
        
     
    return df

def feature_eng(df):
    """Create new features
    
    Parameters
    ----------
    df : DataFrame of the training or test set
        Combine siblings and parents to derive a family count.
        Create a column to determine if the person was alone or not.
        Create a Title column from Names that categorizes the title of the person.
        Bucket ticket fares into four quadrants.
        Bucket ages into 5 parts.
        
    Returns
    -------
    df: DataFrame with three additional features and two variables with bucketed values
    """    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df.loc[df.FamilySize > 1, 'IsAlone'] = 0
    df['Title'] = df.Name.str.extract('([A-Za-z]+)\.')
    df['Title'] = df.Title.replace(['Master', 'Don', 'Rev', 'Dr','Major', 'Lady', 
    'Sir','Col', 'Capt', 'Countess', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df.Title.replace('Mme', 'Mrs')
    df['Title'] = df.Title.replace('Ms', 'Miss')
    df['Title'] = df.Title.replace('Mlle', 'Miss')
    df['Fare'] = pd.qcut(df['Fare'], 4)
    df['Age'] = pd.cut(df['Age'], 5)
    
    return df

def encode(df):
    """Encode (factorize) the data
    
    Parameters
    ----------
    df : DataFrame of the training or test set
        Use LabelEncoder to encode five variables for charting purposes and for the 
        RandomForestClassifier
        
    Returns
    -------
    df: DataFrame with the below five variables now encoded 
    """  
    label = LabelEncoder()
    
    df['Sex'] = label.fit_transform(df['Sex'])
    df['Embarked'] = label.fit_transform(df['Embarked'].astype(str))
    df['Title'] = label.fit_transform(df['Title'])
    df['Age'] = label.fit_transform(df['Age'].astype(str))
    df['Fare'] = label.fit_transform(df['Fare'])
    
    return df









if __name__ == "__main__":
    clean_and_train()
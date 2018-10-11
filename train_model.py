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

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    
    # full_data = pd.concat([train_set, test_set], sort=True)
    
    # data_cleaner = [train_set, test_set]

    # # peak at null values before imputation
    # # print('\n',
    # # "Checking for null values: train & test.\n", 
    # # "=" *20, '\n',
    # # train_set.isnull().sum(), '\n',
    # # "=" *10, '\n',
    # # test_set.isnull().sum(), '\n',
    # # "=" *20)
    
    # # impute and drop training and test sets
    # train_set, test_set = impute_and_drop(train_set, test_set)

    # # confirm imputation and dropped columns
    # # print('\n',
    # # "Age, Embarked, and Fare successfully imputed in the training & test sets.\n",
    # # "Cabin, PassengerId, and Ticket were successfully dropped in the training set.\n", 
    # # "=" *20, '\n',
    # # train_set.isnull().sum(), '\n',
    # # "=" *10, '\n',
    # # test_set.isnull().sum(), '\n',
    # # "=" *20)

    # print(train_set.info(),
    # test_set.info(),
    # train_set.sample(10))

    # print(test_set.Age.head())

    
 
    train_set = (train
        .drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=False)
        .pipe(impute)
        .pipe(feature_eng)
        .pipe(factorize))

    test_set = (test
        .pipe(impute)
        .pipe(feature_eng)
        .pipe(factorize))

    print(train_set.shape, test_set.shape)
    print(train_set.isnull().sum(), train_set.info(), test_set.isnull().sum(), test_set.info())

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
        
#     train_set.drop(['Cabin', 'PassengerId', 'Ticket'], axis=1, inplace=True) # drop from training set only
     
    return df

def feature_eng(df):
    """Create new features
    
    Parameters
    ----------
    df : DataFrame of the training or test set
        Combine siblings and parents to derive a family count.
        Create a column to determine if the person was alone or not.
        Bucket ticket fares into four quadrants.
        Bucket ages into 5 parts
        
    Returns
    -------
    df: DataFrame with four additional features
    """    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1    
    df.loc[df.FamilySize > 1, 'IsAlone'] = 0
#     df['Title'] = df['Name'].str.split(". ", expand=True)[1].str.split(".", expand=True)[0]
    df['FareBin'] = pd.qcut(df['Fare'], 4)
    df['AgeBin'] = pd.cut(df['Age'], 5)
    
    return df

def factorize(df):
    """Encode (factorize) the data
    
    Parameters
    ----------
    df : DataFrame of the training or test set
        Use LabelEncoder to encode four categorical variables for charting purposes
        
    Returns
    -------
    df: DataFrame with four additional encoded variables
    """  
    label = LabelEncoder()
    
    df['Sex_encode'] = label.fit_transform(df['Sex'])
    df['Embarked_encode'] = label.fit_transform(df['Embarked'].astype(str))
    # df['Title_encode'] = label.fit_transform(df['Title'])
    df['AgeBin_encode'] = label.fit_transform(df['AgeBin'].astype(str))
    df['FareBin_encode'] = label.fit_transform(df['FareBin'])
    
    return df



















    # imp = Imputer(missing_values='Nan', strategy='mean', axis=0)

    # clf = SVC()

    # steps = [('imputation', imp),
    #         ('SVM', clf)]

    # pipeline = Pipeline(steps)

    # X_train, X_test, y_train, y_test = train_test_split()

    # pipeline.fit(X_train,  y_train)







if __name__ == "__main__":
    main()
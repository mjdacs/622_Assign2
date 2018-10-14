import pandas as pd 
import subprocess


"""
This script starts with calling pull_data() which uses the subprocess library to kaggle-call the titanic 
dataset. Note: you need your kaggle API key, which can be downloaded from your kaggle account as a json file.

Once the train and test files are pulled, they are sent to the function convert_and_check_data() which has 
a couple of steps to pull and check for proper data:
(1) I used static type checking to make sure the file name being pulled in was a string.
(2) Use pd.read_csv to pull in the training and test sets, assign to variables train and test.
(3) The testing is done in a separate function, and we call confirm_data_exists_and_train_isgreater_than_test.
(4) Since pandas already has a built-in EmptyDataError for blank files with the correct filename, 
    I used assert to make sure the files were greater than 100 values in size.
(5) Lastly, assert was used again to make sure the training set was larger than the test set.
(6) If the tests pass, "success" will print and the data sets will be returned by the pull_data function.
"""


# use subprocess to use a kaggle call from the terminal and get the titanic data
def pull_data():

    command = "kaggle competitions download -c titanic"
    subprocess.check_call(command.split())
    train_csv = "train.csv"
    test_csv = "test.csv"
    train_set, test_set = convert_and_check_data(train_csv, test_csv)
    return train_set, test_set
   

# converts train_set and test_set to pandas DataFrames, runs checks, and returns DataFrames if the checks clear
def convert_and_check_data(train_csv: str, test_csv: str):
    
    train_set = pd.read_csv(train_csv) 
    test_set = pd.read_csv(test_csv) 
    print("=" * 33)
    print("csv files converted to DataFrames")   
    confirm_data_exists_and_train_isgreater_than_test(train_set, test_set)
    return train_set, test_set


def confirm_data_exists_and_train_isgreater_than_test(train_set, test_set):

    assert train_set.size > 100, "Not enough data in training set, check file name or source"
    assert test_set.size > 100, "Not enough data in test set, check file name or source"
    assert train_set.size > test_set.size, "Your training set should be larger than the test set"
    print("=" * 33)
    print("SUCCESS checking DataFrames")
    print("=" * 33)






if __name__ == "__main__":
    pull_data()
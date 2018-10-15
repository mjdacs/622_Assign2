

DATA 622 # hw2

Assigned on September 27, 2018
Due on October 17, 2018 11:59 PM EST
15 points possible, worth 15% of your final grade

    Required Reading

    Read Chapter 5 of the Deep Learning Book Read Chapter 1 of the Agile Data Science 2.0 textbook

    Data Pipeline using Python (13 points total)

    Build a data pipeline in Python that downloads data using the urls given below, trains a random forest model on the training dataset using sklearn and scores the model on the test dataset.

    Scoring Rubric

    The homework will be scored based on code efficiency (hint: use functions, not stream of consciousness coding), code cleaniless, code reproducibility, and critical thinking (hint: commenting lets me know what you are thinking!) Instructions:

    Submit the following 5 items on github. ReadMe.md (see "Critical Thinking") requirements.txt pull_data.py train_model.py score_model.py

More details:

requirements.txt (1 point) This file documents all dependencies needed on top of the existing packages in the Docker Dataquest image from HW1. When called upon using pip install -r requirements.txt , this will install all python packages needed to run the .py files. (hint: use pip freeze to generate the .txt file)

pull_data.py (5 points) When this is called using python pull_data.py in the command line, this will go to the 2 Kaggle urls provided below, authenticate using your own Kaggle sign on, pull the two datasets, and save as .csv files in the current local directory. The authentication login details (aka secrets) need to be in a hidden folder (hint: use .gitignore). There must be a data check step to ensure the data has been pulled correctly and clear commenting and documentation for each step inside the .py file. Training dataset url: https://www.kaggle.com/c/titanic/download/train.csv Scoring dataset url: https://www.kaggle.com/c/titanic/download/test.csv

train_model.py (5 points) When this is called using python train_model.py in the command line, this will take in the training dataset csv, perform the necessary data cleaning and imputation, and fit a classification model to the dependent Y. There must be data check steps and clear commenting for each step inside the .py file. The output for running this file is the random forest model saved as a .pkl file in the local directory. Remember that the thought process and decision for why you chose the final model must be clearly documented in this section. eda.ipynb (0 points)

[Optional] This supplements the commenting inside train_model.py. This is the place to provide scratch work and plots to convince me why you did certain data imputations and manipulations inside the train_model.py file.

score_model.py (2 points) When this is called using python score_model.py in the command line, this will ingest the .pkl random forest file and apply the model to the locally saved scoring dataset csv. There must be data check steps and clear commenting for each step inside the .py file. The output for running this file is a csv file with the predicted score, as well as a png or text file output that contains the model accuracy report (e.g. sklearn's classification report or any other way of model evaluation).

    Critical Thinking (2 points total) Modify this ReadMe file to answer the following questions directly in place.
        Kaggle changes links/ file locations/login process/ file content
			MD- I used the kaggle API, so if links are changed, the new links should be mapped to
			the same command in the command line interface. Howerever the API could be updated 
			and one would need to learn the new commands to pull the data. The folder name could
			also be changed, and you would have to figure out what the new names are.

        We run out of space on HD / local permissions issue - can't save files
			MD- If there are local issues the files will need to be downloaded to a cloud repo or
			storage point and the program should then pull the data from there.

        Someone updated python packages and there is unintended effect (functions retired or act differently)
			MD- This is where a requirements.txt document comes in, so you know what versions
			of depencies were used to ensure a successfully working program.

        Docker issues - lost internet within docker due to some ip binding to vm or local routing issues( I guess this falls under lost internet, but I am talking more if docker is the cause rather then ISP)
			MD- I need to understand Docker a bit better, but it seems like you will have to dig
			into the image that was created and see if there is something that can be switched in
			the Dockerfile to protect against these issues.

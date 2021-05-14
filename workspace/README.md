# Disaster Response Pipeline Project
In this Project Workspace, i used a data set containing real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

The project include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will showed my software skills, including my ability to create basic data pipelines and write clean, organized code!


# About Project in detail
## Data Pipelines: Jupyter Notebooks
I have used Jupyter notebooks in Project Workspaces and used instructions which helped me in getting started with both data pipelines. 

## Project Workspace - ETL
The first part of your data pipeline is the Extract, Transform, and Load process. Here, I read the dataset, clean the data, and then store it in a SQLite database. i did  data cleaning with pandas as suggested in Project Detail. To load the data into an SQLite database, I have used the pandas dataframe .to_sql() method, which I can use with an SQLAlchemy engine.

I even used some exploratory data analysis in order to clean the data set. Though I do not need to submit this exploratory data analysis as part of mine project, it will be neeeded to include your cleaning code in the final ETL script, process_data.py. It was great learning to use project workspace as i could easily communicate with my mentor alsofor any issue.

## Project Workspace - Machine Learning Pipeline
For the machine learning portion, I split the data into a training set and a test set. Then, I have created a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). Finally, I had export mine model to a pickle file. After completing the notebook, I have included mine final machine learning code in train_classifier.py.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `n data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ General
    https://view6914b2f4-3001.udacity-student-workspaces.com/%22(My web address with my space id.

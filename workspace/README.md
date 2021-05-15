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
File Descriptions
There are 2 notebooks available here to showcase work related to the above questions. One for ETL Pipeline to process the data, and the other is ML Pipeline to build the model. The notebooks are exploratory in searching through the data pertaining to the questions showcased by the notebook title. Markdown cells & comments were used to assist in walking through the thought process for individual steps.

## Below is File structure and explanations for files inside it.
app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py
|- classifier.pkl # saved model
README.md


process_data.py: ETL Pipeline Script to process data
ETL Pipeline Preparation.ipynb: jupyter notebook records the progress of building the ETL Pipeline
disaster_messages.csv: Input File 1, CSV file containing messages
disaster_categories.csv: Input File 2, CSV file containing categories
DisasterResponse_Processed.db: Output File, SQLite database, and also the input file of train_classifier.py
In working_directory/models:

train_classifier.py: Machine Learning pipeline Script to fit, tune, evaluate, and export the model to a Python pickle file
ML Pipeline Preparation.ipynb: jupyter notebook records the progress of building the Machine Learning Pipeline
model.p: Output File, a pickle file of the trained Machine Learning Model
In working_directory/app:

templates/*.html: HTML templates for the web app.
run.py: Start the Python server for the web app and prepare visualizations.

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

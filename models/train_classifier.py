import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from sqlalchemy import create_engine

# For machine learning
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import warnings

warnings.simplefilter('ignore')

# For nlp
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import pickle



def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:/// DisasterResponse.db')
    df = pd.read_sql_table('message_categorys', engine) 
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

from nltk.corpus import stopwords
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
def tokenize(text):
    # Detect URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        
        text = text.replace(url, 'urlplaceholder')
    
    # Normalize and tokenize and remove punctuation
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # Lemmatize
    lemmatizer=WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return tokens
def model_pipeline():
    pipeline = Pipeline([(
        'vect', CountVectorizer(tokenizer = tokenize)), 
        ('tfidf', TfidfTransformer()), 
        ('clf',MultiOutputClassifier(RandomForestClassifier()))])
    pipeline.get_params().keys()
    #Split data into train and test sets
    
    return pipeline


def build_model():
    model = model_pipeline()
    parameters = {'vect__min_df': [1, 5],'tfidf__use_idf':[True, False],'clf__estimator__n_estimators':[10, 25], 'clf__estimator__min_samples_split':[2, 5, 10]}
    cv = GridSearchCV(estimator=model, param_grid=parameters, verbose=3)
    
    print('Best Parameters:', cv.param_grid)
    return cv
        
        
        
        
       

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print('Category: {} '.format(category_names[i]))
        print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, Y_pred[:, i])))
        print('F1 {}\n\n'.format(f1_score(Y_test.iloc[:, i].values, Y_pred[:, i],average='weighted')))
        # print(classification_report(Y_test.iloc[:, 1:].values, np.array([x[1:] for x in Y_pred]), target_names = Y.columns))
        # where is F-1 score is low, Lets see the distribution of class
   
    #Improving the model as scoring in above model was not that efficent
    #grid search
    

def save_model(model, model_filepath):
    pickle.dump(cv, open('model_dis.sav', 'wb'))
    loaded_model = pickle.load(open('disaster_model.sav', 'rb'))
    Y_pred = loaded_model.predict(X_test)
    print(classification_report(Y_test.iloc[:, 1:].values, np.array([x[1:] for x in Y_pred]), target_names = Y.columns))

    
   


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
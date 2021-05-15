import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline                import Pipeline, FeatureUnion
from sklearn.multioutput             import MultiOutputClassifier
from sklearn.ensemble                import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection         import train_test_split, GridSearchCV
from sklearn.metrics                 import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from scipy.stats.mstats              import gmean
from sklearn.base                    import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine



app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
#engine = create_engine('sqlite:///../DisasterResponse.db')
#df = pd.read_sql_table('message_categorys', engine)
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_categorys', engine)

# load model
model = joblib.load("../models/classifier.pkl")
#model = joblib.load("../workspace/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    category_names      = df.iloc[:,4:].columns
    category_counts     = (df.iloc[:,4:] != 0).sum().values
    category_percentage = category_counts/category_counts.sum()
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    #graphs = [
       # {
            #'data': [
              #  Bar(
                  #  x=genre_names,
                  #  y=genre_counts
               # )
           # ],

            # 'title': 'Distribution of Message Genres',
               # 'yaxis': {
                 #   'title': "Count"
               # },
               # 'xaxis': {
                #    'title': "Genre"
              #  }
            #}
       # }
    graphs = [
        # Graph 1: Genre Graph
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        # Graph 2: Category Graph
        {
            'data': [
                Pie(
                    labels = category_names,
                    values = category_percentage
                )
            ],

            'layout': { 'title': 'Percentage of Message Categories' }
        },
        
        # Graph 3: Category Graph
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Message Count of each Categories"
                },
                'xaxis': {
                    'title': "Messages Category",
                    'tickangle': 30
                }
            }
        }
        
    ]
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
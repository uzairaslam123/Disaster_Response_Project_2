import sys
import re
import pandas as pd
import nltk
import pickle
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    
    INPUT: 
    database_filepath - The file path to the database
    
    
    OUTPUT: 
    X - The dataset X after being splitted into X and Y for machine leaning model.
    Y - The dataset Y after being splitted into X and Y for machine leaning model.
    category_names - The names of 36 different categories.
    
    This function performs the following.
    1. Loads the dataset into the pandas dataframe 'df'.
    2. Splits the dataset into X and Y dataframes to be used for machine learning model.
    3. Save and return the 36 different category names used for evaluating our machine leanrning model's accuracy.
    
    '''
    
    # load data from database and store it in dataframe 'df'
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df', engine)
    
    # Spliiting the data into X and Y data frames to be used for Machine Leanring model.
    X = df['message']
    Y = df.iloc[:,4:]
    
    #Saving the list of 36 different category names
    category_names = list(df.columns[4:])
    
    return X,Y,category_names
    


def tokenize(text):
    '''
    
    INPUT: 
    text - The text document that will be further cleaned.
    
    
    OUTPUT: 
    clean - The clean text after normalization, tokenization, stor word removal and lemmatization.
    
    This function performs the following.
    1. It removes the puncations and case normalizese the input text.
    2. It tokenizese the text.
    3. It removes the stop words.
    4. It performs lemmatization on text and also removes any unnecessary white spaces.
    5. In the end it outputs a cleaned text.
    
    '''
    
    #Punctuation removal and case normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #tokenize text
    tokens = word_tokenize(text)
    
    #Stop word removal 
    words = [w for w in tokens if w not in stopwords.words("english")]
    
    #initiate Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #Performs lemmatization and also strips off any unnecessary white spaces
    clean = [lemmatizer.lemmatize(w, pos='n').strip() for w in words]
    clean = [lemmatizer.lemmatize(w, pos='v').strip() for w in clean]
    
    return clean

def build_model():
    '''
    
    INPUT: 
    There is no input.
    
    
    OUTPUT: 
    model - The GridSearch output.
    
    This function performs the following.
    1. It creates a pipeline with the right estimators and transformers.
    2. It creates a model Machine Learning pipeline based on RandomForestClassifier which uses MultiOutputClassifier
    for prediciting multiple target variables. This model uses Grid Search approach for finding the best set of parameters.
    3. Creates and returns a GridSearchCV model. 
    
    '''
    #creating a Machine Learning pipline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])
    
    #defin parameters for GridSearchCV
    parameters = {'clf__estimator__min_samples_split': [4, 5],
              'clf__estimator__n_estimators': [20, 30],
              'clf__estimator__criterion': ['entropy', 'gini']
             }
    #saving the GridSearchCV model
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    
    INPUT: 
    model - The GridSearchCV model that was created before.
    X_test - The test dataset which will be used for evaluation.
    Y_test - The set of True labels.
    category_names - It contains the names of labels for 36 label.
    
    
    OUTPUT: 
    The output is the accuracy and classification report for each category.
    
    This function performs the following.
    1. It predicts on the test data (X_test) and saves it in Y_pred.
    2. It loops over all the categories and prints classification report and accuracy details.
    
    '''
    #predict on the the test data (X_test) using the model
    Y_pred = model.predict(X_test)
    
    #loop over all the categories and print classification report and accuracy
    for category in range(len(category_names)):
        print("Category:", category_names[category], "\n", classification_report(Y_test.iloc[:, category].values,
                                                                                 Y_pred[:,category]))
        print('Accuracy of %25s: %.2f' %(category_names[category], accuracy_score(Y_test.iloc[:,category].values,
                                                                                  Y_pred[:,category])))


def save_model(model, model_filepath):
    '''
    INPUT: 
    model - The GridSearchCV model that was created before.
    model_filepath - The filepath where the model should be saved
    
    
    OUTPUT: 
    A pickle file of the saved model.
    
    This function performs the following.
    1. It saves the model as the pickle file at the given filepath.
    
    '''
    
    pickle.dump(model, open(model_filepath, "wb"))
    
    

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
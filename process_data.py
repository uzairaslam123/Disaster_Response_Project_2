import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    '''
    INPUT: 
    messages_filepath - The file path to the 'messages' dataset
    categories_filepath - The file path to the 'categories' dataset
    
    
    OUTPUT: 
    df - Merged dataframe
    
    This function loads the 'messages' and 'categories' dataset and returns a merged dataframe
    '''
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    #load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='outer', on='id')
    
    return df


def clean_data(df):
    
    '''
    INPUT: 
    df - The combined dataframe 
    
    OUTPUT: 
    df - Clean dataframe 
    
    This function receives the combined dataframe 'df' and performs the following data cleaning operations on it
    1. Split the 'categories' column in separate column for each category.
    2. Splitting is performed on the ';' character.
    3. After splitting, the original 'categories' column is broken down into 36 different categories.
    4. Name the newly created columns with the appropriate column names where each column name represents a category name.
    5. Convert the 'category' values from string values e.g from 'related_0', 'requested_1' to 0 & 1 only.
    6. Replace the original 'category' column in the dataframe 'df' with the newly created 'caetgory' columns.
    7. In the end remove the duplicate values.
    
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Convert the category values to 0 or 1. 
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, )
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
    INPUT: 
    df - The cleaned dataframe to be saved.
    database_filename - The database filename where the dataframe 'df' will be saved
    
    
    OUTPUT: 
    SQLite database
    
    This function is used to save the cleaned the dataframe 'df' to the database.
    '''
    
    #Establish a connection to the database
    engine = create_engine('sqlite:///' + database_filename)
    
    #write the data in 'df' to sql database 
    df.to_sql('df', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
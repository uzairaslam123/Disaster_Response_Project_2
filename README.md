### Table of Contents

1. [Introduction](#introduction)
2. [File Descriptions](#descriptions)
3. [Installation and Dependencies](#installation)
4. [How to setup and run](#setup)
5. [Conclusion](#conclusion)
6. [Licensing, Authors, and Acknowledgments](#licensing)

## Introduction<a name="introduction"></a>
# Disaster_Response_Project_2
This is the second project of my Udacity Data Science Nano Degree. This project deals with diasters like, earthquake, floods and storms etc. and the response from the people in such situations. Social media is the most widely used mode of information and communication in such events and it can be used to understand the gravity of the situation. For example, Twitter gets flooded with new messages every minute about the latest developments in these events and the idea of this project is to use such messages from Twitter, to analyze each message and classify it into a particular categories. This can then be used by the authorities to better utilize their resources in the most efficient way and to meet the requirments without any delays. 

To be able to classifly Twitter messages, we would have to use Natural Language Proccessing (NLP) to analyze the message text and then use the Multi-Classfication Machine Learning models to classify each message into the appropriate category. 
The data set that we will be using in this project has been provided by Figure Eight Inc.


## File Descriptions<a name="descriptions"></a>

Below is the file structure of this project.

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md

- The **app** folder contains html templates for the app and a python script **run.py** that creates and runs the web app.

- The **data** folder contains the **process_data.py** python file whihc is an ETL (Extract, Transport and Load) pipeline that processes the data provided in the files **categories.csv** and **messages.csv**  and saves it in the database **DisasterResponse**.db. 

- The **models** folder contains a **train_classifier.py** python file which is a Machine Leanring pipeline to fetch the data from the database and trains the model and then finall saves the trained model with the best hyperparameters in the pickle file named **classifier.pkl**.



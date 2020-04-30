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

```bash
├── README.md
├── app
│   ├── run.py # runs the web app
│   └── templates 
│       ├── go.html # classificaton page
│       └── master.html # home pagee with visualisations
├── data
│   ├── DisasterResponse.db # database with clean data
│   ├── disaster_categories.csv # category dataset
│   ├── disaster_messages.csv # messages dataset
│   └── process_data.py # process and clean data script
├── notebooks
│   ├── ETL\ Pipeline\ Preparation.ipynb # ETL notebook
│   └── ML\ Pipeline\ Preparation.ipynb # ML notebook
└── models
    ├── classifier.pkl # model
    ├── train_classifier.py # model training script
```   

- The **app** folder contains html templates for the app and a python script **run.py** that creates and runs the web app.

- The **data** folder contains the **process_data.py** python file whihc is an ETL (Extract, Transport and Load) pipeline that processes the data provided in the files **categories.csv** and **messages.csv**  and saves it in the database **DisasterResponse.db**.

- The **notebooks** folder contains the juptyter notebook files **ETL\ Pipeline\ Preparation.ipynb** and **ML\ Pipeline\ Preparation.ipynb** that were used to write the ETL and ML piplines for the data and to perform some further analysis 

- The **models** folder contains a **train_classifier.py** python file which is a Machine Leanring pipeline to fetch the data from the database and trains the model and then finall saves the trained model with the best hyperparameters in the pickle file named **classifier.pkl**.



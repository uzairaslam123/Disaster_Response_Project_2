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

- The **models** folder contains a **train_classifier.py** python file which is a Machine Leanring pipeline to fetch the data from the database and trains the model and then finally saves the trained model, with the best hyperparameters, in the pickle file named **classifier.pkl**.


## Installation and Dependencies<a name="Installation"></a>

To be able to run and use this web app, you can either clone or download this repository to your local machine.
The next step will be to open a new terminal window. You should be in the home folder, but if not, then use terminal commands to navigate inside the folder with the run.py file.

Type in the command line:

```bash 
cd app/ #to move to the app folder
python run.py #command to run the web app
```
Normally the first step would be to run the data pre-processing file **process_data.py** by using the command 

```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db #run the ETL pipeline and saves the data in the database

```
After this step, Machine Learnign pipeline should be run, to fetch the data from the database and train the model and then finally save it as a pickle file. This can be done by the following command.

```bash
python models/train_classifier.py data/DisasterResponse.db models/ada_classifier.pkl #command to run the machine learnign pipeline and train the model.

```

#### Dependencies

Following packages and python libraries were used in this project:
- pandas
- re
- numpy
- nltk
- pickle
- scikit-learn
- plotly
- flask
- sqlalchemy

## How to setup and run<a name="setup"></a>

Once you have successfully installed all the files, you can use the web app by visiting http://0.0.0.0:3001/.
There you willfind three different visual representations of the data which will help you understand the app and its functionality better. 

You will also find a text box, where you can enter any text message and the model will classify it into 36 different categories. 

## Conclusion<a name="conclusion"></a>
In the end i would like to say that it is nearly impossible to avoid any of the natural diasters but we can use technology to improve our response to such calamities. Machine Learning has proven to be beneficial in all areas of life and using Machine Learning in thiis project, we can close the gap between the victims of natural disasters and the people and organizations who are trying to save such victims. 

The questions is, how can you use technology to make this world a better place?


## Licensing, Authors, and Acknowledgments<a name="licensing"></a>

This project was made possible by Figure Eight Inc. for providing the dataset and Udacity, for providing us such a great platform for learning.
Feel free to use my code in any of your projects

Enjoyyy.



# Question-Answering-System

Simple Question Answering System implementation using BERT. 

## Required

This repository requires Python 3.7

```
git clone https://github.com/stephenwilliams1990/Question-Answering-System.git
cd Question-Answering-System
```
Install the required packages
```
python -m pip install -r requirements.txt
```

## Usage

1. Download sample data from this repository. Or, alternatively you can upload your own database to use. Just ensure to have it in the same format as the sample database provided here with question and answers in a dataframe. Then either name your database the same as it is named in this repository, or alternatively update the qaGeneratorBert.py file to change the name of the csv file read in by the code. 
2. Open a terminal and run the following code to use this Question Answering system.

```
python qaGeneratorBert.py
```
**NOTE:** This code is a simple version of a Question Answering system that can be run directly from the command line.

## A Note on included data in this reposity

A dummy database has been provided in this repository for sample testing. However this can be used on any Question/Answer database as desired.


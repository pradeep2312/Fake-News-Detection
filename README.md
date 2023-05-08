# Fake News Detection Project Work


## Description of the project 
Fake news analysis project is the process of detecting and categorizing news articles that contain misinformation or fabricated information. The project involves using machine learning techniques, particularly RNN models with LSTM layers, to train a model on a dataset of news articles labeled as either fake or real. The model is trained to identify patterns in the language and context of the articles and use those patterns to classify new articles as either real or fake. The project typically involves steps such as data preprocessing, model building, training and evaluation, and deployment for use in real-world applications. The ultimate goal of the project is to provide an application to help them identify and combat fake news.

## Datasets: 

Analysis is performed using two different datasets such as [True Data](https://github.com/pradeep2312/Fake-News-Detection/blob/master/Dataset/True.csv) and [Fake Data](https://github.com/pradeep2312/Fake-News-Detection/blob/master/Dataset/Fake.csv) and these datasets are gathered from the Kaggle.

## Modelling :

In this project we have developed a sequential RNN model with three layers as embedding layer , LSTM layer , and dense layer is created for fake news analysis.

**Embedding layer:** Maps input words to a vector space, where similar words are closer together.

**LSTM layer:** Processes sequential input data and learns long-term dependencies by selectively remembering or forgetting information over time.

**Dense layer:** Performs a linear transformation on the input data to project it into a higher-dimensional space.

## Result Analysis: 

The model is trained and tested thoroughly and model is successfully classifying the text with an accuracy score of %.

## Application : 

A web application was developed using the Streamlit framework. The trained model was saved as a pickle file and integrated into the front-end to enable user input and prediction. The web application allows users to interact with the model and obtain predictions on their input data.




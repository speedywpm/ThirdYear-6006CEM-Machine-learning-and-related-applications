# Car Insurance Claim Prediction

This is a machine learning project that aims to predict whether an individual is likely to make a car insurance claim. The project uses historical car insurance claims data and builds a model to predict the likelihood of a claim being made by an individual based on their demographic and vehicle information.

## Table of Contents
1. Installation
2. Usage
3. Dataset
4. Methodology
5. Results

## Installation

The project is implemented in a Jupyter Notebook and requires the following Python libraries:

* pandas
* numpy
* seaborn
* matplotlib
* scikit-learn
To install the required libraries, run:

`pip install pandas numpy seaborn matplotlib scikit-learn`

## Usage

To run the project, open the Jupyter Notebook and run all the cells. This will load the dataset, preprocess the data, train the models, and evaluate the performance of the models. The final cell will display a summary of the evaluation results.

## Dataset

Dataset: https://www.kaggle.com/datasets/sagnik1511/car-insurance-data
The dataset used for this project is named Car_Insurance_Claim.csv. This dataset contains the following features:

* ID: Unique identifier for each individual
* GENDER: Gender of the individual
* VEHICLE_TYPE: Type of vehicle owned by the individual
* VEHICLE_YEAR: Year of the vehicle's manufacture
* AGE: Age of the individual
* DRIVING_EXPERIENCE: Number of years the individual has been driving
* EDUCATION: Highest level of education attained by the individual
* INCOME: Annual income of the individual
* RACE: Race of the individual
* POSTAL_CODE: Postal code of the individual's address
* CREDIT_SCORE: Credit score of the individual
* ANNUAL_MILEAGE: Annual mileage driven by the individual
* CLAIM_INDICATOR: Binary value indicating whether the individual has made a claim (1) or not (0)

## Methodology

The methodology of this project involves the following steps:

1. Load and preprocess the dataset: The dataset is loaded into a pandas DataFrame, and irrelevant columns (ID and POSTAL_CODE) are dropped. Missing values are imputed using the mean value of the respective columns.
2. Encode categorical variables: The categorical variables are encoded using LabelEncoder from scikit-learn.
3. Split the dataset: The dataset is split into training (60%) and testing (40%) sets.
4. Standardize the features: The features are standardized using StandardScaler from scikit-learn.
5. Train and evaluate multiple machine learning models: The following models are trained and evaluated:
   - Logistic Regression
   - K-Nearest Neighbors Classifier
   - Neural Network (Multilayer Perceptron Classifier)
6. Evaluation metrics: The performance of each model is evaluated using the following metrics:
   - Accuracy
   - Confusion matrix
   - Classification report
   - Cross-validation
   
## Results

The results of the evaluation are displayed in a simplified view at the end of the Jupyter Notebook, including accuracy, confusion matrix, classification report, and cross-validation results for each model.

The results of the evaluation indicate that the models have varying levels of performance. It is recommended to choose the model with the best performance based on the evaluation metrics and the specific needs of the project.

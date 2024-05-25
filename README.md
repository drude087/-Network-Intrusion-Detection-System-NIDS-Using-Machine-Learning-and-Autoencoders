# Network-Intrusion-Detection-System-NIDS-Using-Machine-Learning-and-Autoencoders
**Project Description**
This project implements a Network Intrusion Detection System (NIDS) using multiple machine learning models and an autoencoder neural network. The goal of this system is to analyze network traffic data and accurately identify potential cyber attacks. The project involves data preprocessing, feature selection, model training, and evaluation, as well as a prediction script that can be used to classify new network traffic data.

**Features**

**Data Preprocessing:** Clean and preprocess network traffic data to ensure high-quality input for model training.

**Feature Selection:** Use Recursive Feature Elimination (RFE) and Random Forest feature importance to select the most relevant features.

**Model Training:** Train multiple machine learning models including Random Forest, Naive Bayes, Decision Tree, K-Neighbors, Logistic Regression, and an Autoencoder.

**Model Evaluation:** Evaluate the performance of each model using cross-validation and various metrics such as accuracy, confusion matrix, and classification report.

**Prediction:** Use trained models to predict and classify new network traffic data as either an attack or normal traffic.

**Datasets:**

The project uses a collection of CSV files containing network traffic data from various scenarios including normal operations and different types of cyber attacks. These datasets are concatenated and cleaned to form a single comprehensive dataset for training and evaluation.

**Implementation Details**

**Data Loading and Cleaning:** Load multiple CSV files, concatenate them into a single DataFrame, drop any missing values, and strip whitespace from column names.

**Feature Selection:** Select relevant columns and apply feature importance techniques to identify the most significant features for model training.

**Data Splitting:** Split the cleaned data into training and testing sets.

**Standardization:** Apply standard scaling to the numeric features to ensure all features contribute equally to the model training.

**One-Hot Encoding:** Encode the target labels using OneHotEncoder.

**Model Training:** Train various machine learning models and an autoencoder neural network using the selected features and encoded labels.

**Model Evaluation:** Evaluate each model using cross-validation, calculate accuracy, confusion matrix, and generate a classification report.

**Model Saving:** Save the trained models and preprocessing objects using joblib and TensorFlow's model saving functionality.

**Prediction Script:** Provide a separate script to load the saved models and preprocessing objects, preprocess new data, and make predictions indicating whether each sample is an attack or normal traffic.

**Files:**

model.py: Script for training the models and saving them.

main.py: Script for loading the saved models and making predictions on new data.

scaler.pkl, onehot_encoder.pkl, random_forest_model.pkl, naive_bayes_classifier_model.pkl, decision_tree_classifier_model.pkl, kneighbors_classifier_model.pkl, logistic_regression_model.pkl, autoencoder_model.h5: Saved preprocessing objects and trained models.

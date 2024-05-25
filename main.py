import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Load the saved models and preprocessing objects
scaler = joblib.load('scaler.pkl')
onehotencoder = joblib.load('onehot_encoder.pkl')
random_forest_model = joblib.load('random_forest_model.pkl')
naive_bayes_model = joblib.load('naive_bayes_classifier_model.pkl')
decision_tree_model = joblib.load('decision_tree_classifier_model.pkl')
kneighbors_model = joblib.load('kneighbors_classifier_model.pkl')
logistic_regression_model = joblib.load('logistic_regression_model.pkl')
autoencoder_model = tf.keras.models.load_model('autoencoder_model.h5')

# Function to preprocess new data
def preprocess_data(df):
    selected_columns = [
        "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Subflow Fwd Packets",
        "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_backward",
        "act_data_pkt_fwd", "min_seg_size_forward", "Label"
    ]
    df = df[selected_columns]
    df = df.dropna()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    sc_data = scaler.transform(df[numeric_cols])
    sc_df = pd.DataFrame(sc_data, columns=numeric_cols)
    return sc_df

# Function to make predictions
def predict_attack(df):
    preprocessed_data = preprocess_data(df)
    models = {
        'Random Forest': random_forest_model,
        'Naive Bayes': naive_bayes_model,
        'Decision Tree': decision_tree_model,
        'KNeighbors': kneighbors_model,
        'Logistic Regression': logistic_regression_model,
        'Autoencoder': autoencoder_model
    }
    results = {}
    for name, model in models.items():
        if name == 'Autoencoder':
            preds = model.predict(preprocessed_data)
            preds = np.argmax(preds, axis=1)  # Get the index of the max value as prediction
        else:
            preds = model.predict(preprocessed_data)
        results[name] = preds
    return results

# Function to print predictions
def print_predictions(predictions):
    for model_name, preds in predictions.items():
        print(f"\nPredictions using {model_name}:")
        for i, pred in enumerate(preds):
            if pred == 1:
                print(f"Sample {i}: It's an attack")
            else:
                print(f"Sample {i}: No attack")

# Example usage
if __name__ == "__main__":
    # Load new data
    new_data = pd.read_csv('path_to_new_data.csv')

    # Make predictions
    predictions = predict_attack(new_data)

    # Print predictions
    print_predictions(predictions)

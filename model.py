import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import joblib

# Display all columns and rows in the dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Read datasets
nrows = 20000
datasets = [
    "Friday-WorkingHours-AfternoonDDos.pcap_ISCX.csv",
    "Friday-WorkingHours-AfternoonPortScan.pcap_ISCX.csv",
    "Friday-WorkingHoursMorning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-AfternoonInfilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-MorningWebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv"
]

dfs = [pd.read_csv(f"dataset/{dataset}", nrows=nrows) for dataset in datasets]
df = pd.concat(dfs)
del dfs

# Data cleaning and preparation
df = df.dropna()
df.columns = df.columns.str.strip()

# Select relevant columns
selected_columns = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Label"
]
df = df[selected_columns]

# Split dataset into train and test sets
train, test = train_test_split(df, test_size=0.3, random_state=10)

# Standardize the data
scaler = StandardScaler()
numeric_cols = train.select_dtypes(include=['float64', 'int64']).columns
sc_train = scaler.fit_transform(train[numeric_cols])
sc_test = scaler.transform(test[numeric_cols])

sc_traindf = pd.DataFrame(sc_train, columns=numeric_cols)
sc_testdf = pd.DataFrame(sc_test, columns=numeric_cols)

# One-hot encode the labels
onehotencoder = OneHotEncoder()
trainDep = onehotencoder.fit_transform(train[['Label']]).toarray()
testDep = onehotencoder.transform(test[['Label']]).toarray()

train_X = sc_traindf
train_y = trainDep[:, 0]
test_X = sc_testdf
test_y = testDep[:, 0]

# Random Forest feature importance
rfc = RandomForestClassifier()
rfc.fit(train_X, train_y)
score = np.round(rfc.feature_importances_, 3)
importances = pd.DataFrame({'feature': train_X.columns, 'importance': score})
importances = importances.sort_values('importance', ascending=False).set_index('feature')

plt.rcParams['figure.figsize'] = (11, 4)
importances.plot.bar()
plt.show()

# Recursive Feature Elimination
rfe = RFE(rfc, n_features_to_select=20)
rfe = rfe.fit(train_X, train_y)
selected_features = train_X.columns[rfe.get_support()]

# Update train and test sets with selected features
train_X = train_X[selected_features]
test_X = test_X[selected_features]

# Split the train set again for model training and evaluation
X_train, X_val, Y_train, Y_val = train_test_split(train_X, train_y, train_size=0.70, random_state=2)

# Model initialization and training
models = [
    ('Naive Bayes Classifier', BernoulliNB()),
    ('Decision Tree Classifier', tree.DecisionTreeClassifier(criterion='entropy', random_state=0)),
    ('KNeighbors Classifier', KNeighborsClassifier(n_jobs=-1)),
    ('Logistic Regression', LogisticRegression(n_jobs=-1, random_state=0))
]

for name, model in models:
    model.fit(X_train, Y_train)
    scores = cross_val_score(model, X_train, Y_train, cv=10)
    accuracy = metrics.accuracy_score(Y_train, model.predict(X_train))
    confusion_matrix = metrics.confusion_matrix(Y_train, model.predict(X_train))
    classification = metrics.classification_report(Y_train, model.predict(X_train))
    
    print(f"\n============================== {name} Model Evaluation ==============================\n")
    print(f"Cross Validation Mean Score:\n {scores.mean()}\n")
    print(f"Model Accuracy:\n {accuracy}\n")
    print(f"Confusion matrix:\n {confusion_matrix}\n")
    print(f"Classification report:\n {classification}\n")

# Model evaluation on test set
for name, model in models:
    accuracy = metrics.accuracy_score(Y_test, model.predict(test_X))
    confusion_matrix = metrics.confusion_matrix(Y_test, model.predict(test_X))
    classification = metrics.classification_report(Y_test, model.predict(test_X))
    
    print(f"\n============================== {name} Model Test Results ==============================\n")
    print(f"Model Accuracy:\n {accuracy}\n")
    print(f"Confusion matrix:\n {confusion_matrix}\n")
    print(f"Classification report:\n {classification}\n")

# Define and train autoencoder model
def getModel():
    inp = Input(shape=(X_train.shape[1],))
    d1 = Dropout(0.3)(inp)
    encoded = Dense(8, activation='relu', activity_regularizer=regularizers.l1(10e-5))(d1)
    decoded = Dense(X_train.shape[1], activation='relu')(encoded)
    autoencoder = Model(inp, decoded)
    autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return autoencoder

autoencoder = getModel()
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
history = autoencoder.fit(X_train, Y_train, epochs=32, batch_size=150, shuffle=True, validation_split=0.1, callbacks=[callback])

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Training and validation loss')
plt.xlabel('epoch')
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Training and validation accuracy')
plt.xlabel('epoch')
plt.show()

# Save the models and scaler
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(onehotencoder, 'onehot_encoder.pkl')
joblib.dump(rfc, 'random_forest_model.pkl')
for name, model in models:
    joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")

# Save the autoencoder model
autoencoder.save('autoencoder_model.h5')

# Keras
import keras
from keras import regularizers
from keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Other  
import librosa
import librosa.display
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import seaborn as sns
import glob 
import os
import pickle
import IPython.display as ipd  # To play sound in the notebook
import numpy as np

# Load the meta-data file
ref = pd.read_csv("./Data_path.csv") # might need revision for location
ref.head()

# Initialize DataFrame for features
df = pd.DataFrame(columns=['feature'])

# Loop feature extration over the dataset
for index, path in enumerate(ref.path):
    X, sample_rate = librosa.load(path, res_type='kaiser_fast', duration=2.5, sr=44100, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    df.loc[index] = [mfccs]

# Combine features with meta-data
df = pd.concat([ref, pd.DataFrame(df['feature'].values.tolist())], axis=1)
df = df.fillna(0)

# Split the data into training and testing sets
X = df.drop(['path', 'labels', 'source'], axis=1)
y = df.labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)

# Normalize the data
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Covert data format to numpy arrays and one-hot encode the target
X_train = np.array(X_train)
X_test = np.array(X_test)
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train))
y_test = to_categorical(lb.fit_transform(y_test))

# Pickle the label encoder for future use
with open('labels', 'wb') as outfile:
    pickle.dump(lb, outfile)

# Expand the dimensions for CNN input
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)


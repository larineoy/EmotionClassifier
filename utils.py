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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

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

import sounddevice as sd
from scipy.io.wavfile import write

def predict(audio_recording):

    train_data = pd.read_csv("./train_features.csv")

    scaler = StandardScaler()
    scaler.fit(train_data.drop(["label", "name"], axis=1))
    x_train = scaler.transform(train_data.drop(["label", "name"], axis=1))

    pca = PCA(n_components=75)
    train_pca = pca.fit_transform(x_train)

    # Recording the audio from user's microphone
    # sample_rate = 22050  # Sample rate
    # seconds = 3  # Duration of recording

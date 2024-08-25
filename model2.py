import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Activation, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam

# Function to extract MFCC features
def extract_features(file_name):
    y, sr = librosa.load(file_name, res_type='kaiser_best', duration=2.5, sr=22050, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=0)
    return mfccs

# Path to datasets
RAVDESS_DIR = './Data/RAVDESS'
TESS_DIR = './Data/TESS'

# Initialize lists to hold features and labels
features = []
labels = []

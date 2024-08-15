# Import libraries 
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import glob 
from sklearn.metrics import confusion_matrix
import IPython.display as ipd
import os
import sys
import warnings
# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

base_path = os.path.dirname(os.path.abspath(__file__))  # Path of the current script
data_path = os.path.join(base_path, 'Data')

TESS = os.path.join(data_path, 'TESS')
RAV = os.path.join(data_path, 'RAVDESS')
SAVEE = os.path.join(data_path, 'SAVEE')
CREMA = os.path.join(data_path, 'CREMA-D')

# Process SAVEE dataset
dir_list = os.listdir(SAVEE)
emotion = []
path = []
for i in dir_list:
    if i[-8:-6] == '_a':
        emotion.append('male_angry')
    elif i[-8:-6] == '_d':
        emotion.append('male_disgust')
    elif i[-8:-6] == '_f':
        emotion.append('male_fear')
    elif i[-8:-6] == '_h':
        emotion.append('male_happy')
    elif i[-8:-6] == '_n':
        emotion.append('male_neutral')
    elif i[-8:-6] == 'sa':
        emotion.append('male_sad')
    elif i[-8:-6] == 'su':
        emotion.append('male_surprise')
    else:
        emotion.append('male_error')
    path.append(os.path.join(SAVEE, i))
SAVEE_df = pd.DataFrame({'labels': emotion, 'source': 'SAVEE', 'path': path})

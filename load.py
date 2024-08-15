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

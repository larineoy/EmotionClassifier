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

# Process RAVDESS dataset
dir_list = os.listdir(RAV)
dir_list.sort()

emotion = []
gender = []
path = []

print("Processing RAVDESS files...")
for f in dir_list:
    file_path = os.path.join(RAV, f)
    if os.path.isfile(file_path) and f.endswith('.wav'):
        part = f.split('.')[0].split('_')
        if len(part) == 4: 
            try:
                actor_id = int(part[0])
                actor_gender = "female" if actor_id % 2 == 0 else "male"
                emotion_code = part[2]

                emotion_map = {
                    "NEU": "neutral", "CAL": "calm", "HAP": "happy",
                    "SAD": "sad", "ANG": "angry", "FEA": "fearful",
                    "DIS": "disgust", "SUR": "surprised"
                }

                emotion_label = emotion_map.get(emotion_code, "Unknown")

                gender.append(actor_gender)
                emotion.append(f"{actor_gender}_{emotion_label}")
                path.append(file_path)
                print(f"Processed file: {file_path}")

            except ValueError:
                print(f"Skipping file due to parsing error: {f}")
        else:
            print(f"Skipping file due to unexpected filename format: {f}")

RAV_df = pd.DataFrame({'labels': emotion, 'source': 'RAVDESS', 'path': path})

# Process TESS dataset
dir_list = os.listdir(TESS)
dir_list.sort()
path = []
emotion = []
for i in dir_list:
    fname = os.listdir(os.path.join(TESS, i))
    for f in fname:
        if i.startswith('OAF_angry') or i.startswith('YAF_angry'):
            emotion.append('female_angry')
        elif i.startswith('OAF_disgust') or i.startswith('YAF_disgust'):
            emotion.append('female_disgust')
        elif i.startswith('OAF_Fear') or i.startswith('YAF_fear'):
            emotion.append('female_fear')
        elif i.startswith('OAF_happy') or i.startswith('YAF_happy'):
            emotion.append('female_happy')
        elif i.startswith('OAF_neutral') or i.startswith('YAF_neutral'):
            emotion.append('female_neutral')
        elif i.startswith('OAF_Pleasant_surprise') or i.startswith('YAF_pleasant_surprised'):
            emotion.append('female_surprise')
        elif i.startswith('OAF_Sad') or i.startswith('YAF_sad'):
            emotion.append('female_sad')
        else:
            emotion.append('Unknown')
        path.append(os.path.join(TESS, i, f))

TESS_df = pd.DataFrame({'labels': emotion, 'source': 'TESS', 'path': path})

# Process CREMA-D dataset
dir_list = os.listdir(CREMA)
dir_list.sort()

gender = []
emotion = []
path = []

print("Processing CREMA-D files...")
# List of female actors by their number
female_actors = set(range(1002, 1100))

for actor_folder in dir_list:
    actor_folder_path = os.path.join(CREMA, actor_folder)
    if not os.path.isdir(actor_folder_path):
        print(f"Skipping non-directory: {actor_folder_path}")
        continue

    for i in os.listdir(actor_folder_path):
        file_path = os.path.join(actor_folder_path, i)
        if i.startswith('.') or not i.endswith('.wav'):
            print(f"Skipping hidden or non-wav file: {i}")
            continue

        part = i.split('-')
        if len(part) < 7:
            print(f"Skipping file due to unexpected filename format: {i}")
            continue

        try:
            emotion_code = int(part[2])
            actor_number = int(part[6].split('.')[0])  # Actor ID is the last part before file extension
            if actor_number in female_actors:
                gender.append('female')
            else:
                gender.append('male')

            # Map the emotion code to the corresponding emotion
            if emotion_code == 1:
                emotion.append(f'male_neutral')
            elif emotion_code == 2:
                emotion.append(f'male_calm')
            elif emotion_code == 3:
                emotion.append(f'male_happy')
            elif emotion_code == 4:
                emotion.append(f'male_sad')
            elif emotion_code == 5:
                emotion.append(f'male_angry')
            elif emotion_code == 6:
                emotion.append(f'male_fearful')
            elif emotion_code == 7:
                emotion.append(f'male_disgust')
            elif emotion_code == 8:
                emotion.append(f'male_surprised')
            else:
                emotion.append('Unknown')

            path.append(file_path)
            print(f"Processed file: {file_path}")

        except ValueError:
            print(f"Skipping due to parsing error in file: {i}")
            continue

CREMA_df = pd.DataFrame({'labels': emotion, 'source': 'CREMA', 'path': path})

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

# Load RAVDESS dataset
for dirname, _, filenames in os.walk(RAVDESS_DIR):
    for filename in filenames:
        if filename.endswith(".wav"):
            file_path = os.path.join(dirname, filename)
            emotion = filename.split('-')[2]  # Extract emotion from filename
            features.append(extract_features(file_path))
            labels.append(emotion)

# Load TESS dataset
for dirname, _, filenames in os.walk(TESS_DIR):
    for filename in filenames:
        if filename.endswith(".wav"):
            file_path = os.path.join(dirname, filename)
            emotion = filename.split('_')[2]  # Extract emotion from filename
            features.append(extract_features(file_path))
            labels.append(emotion)

features_df = pd.DataFrame(features)
features_df['label'] = labels

# Encode the labels
lb = LabelEncoder()
features_df['label'] = lb.fit_transform(features_df['label'])

# Split the data into training and testing sets
X = features_df.iloc[:, :-1].values
y = features_df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Encode the labels to one-hot format
y_train_lb = to_categorical(lb.fit_transform(y_train))
y_test_lb = to_categorical(lb.fit_transform(y_test))

# Check out the data
# print(f'X_train shape: {X_train.shape}')
# print(f'y_train shape: {y_train_lb.shape}')
# print(f'X_test shape: {X_test.shape}')
# print(f'y_test shape: {y_test_lb.shape}')

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for 1D CNN
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Build sequential CNN
model = Sequential()

# Build first layer
model.add(Conv1D(16, 5, padding='same', input_shape=(X_train.shape[1], 1), activation='relu'))

# Build second layer
model.add(Conv1D(32, 5, padding='same', activation='relu'))

# Build third layer
model.add(Conv1D(64, 5, padding='same', activation='relu'))

# Build fourth layer
model.add(Conv1D(128, 5, padding='same', activation='relu'))

# Add dropout
model.add(Dropout(0.1))

# Flatten 
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(lb.classes_), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='upper right')
plt.title('Model Loss')
plt.show()

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=lb.classes_, yticklabels=lb.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_true, y_pred_classes, target_names=lb.classes_))


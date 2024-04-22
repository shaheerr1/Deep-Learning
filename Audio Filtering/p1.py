from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


import joblib


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)


def load_data(data_directory):
    labels = []
    features = []

    for folder in os.listdir(data_directory):
        if not folder.startswith('.'):

            emotion = folder.split('_')[-1]
            folder_path = os.path.join(data_directory, folder)

            for file in os.listdir(folder_path):
                if file.lower().endswith('.wav'):
                    file_path = os.path.join(folder_path, file)
                    feature = extract_features(file_path)
                    features.append(feature)
                    labels.append(emotion)

    return np.array(features), np.array(labels)


features, labels = load_data(
    'C:\datasets\Audio\TESS Toronto emotional speech set data')

print(labels)


label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)


X_train, X_temp, y_train, y_temp = train_test_split(
    features, encoded_labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)


input_shape = (X_train.shape[1], 1)

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(np.unique(encoded_labels)), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


# print(model.summary())

X_train_reshaped = X_train[..., np.newaxis]
X_val_reshaped = X_val[..., np.newaxis]


history = model.fit(X_train_reshaped, y_train, epochs=30,
                    batch_size=64, validation_data=(X_val_reshaped, y_val))


model.save('model.h5')

joblib.dump(label_encoder, 'label_encoder.joblib')

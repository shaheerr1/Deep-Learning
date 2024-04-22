import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import librosa
from tensorflow.keras.models import load_model


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)


label_encoder = joblib.load('label_encoder.joblib')

model = load_model('model.h5')
file_path = 'C:\datasets\OAF_base_fear.wav'


features = extract_features(file_path)
features = np.array(features).reshape(1, -1, 1)

prediction = model.predict(features)
predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

print(f"The predicted label is: {predicted_label[0]}")

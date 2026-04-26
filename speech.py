import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfcc.T

def train_speaker_models(data_path):
    speaker_models = {}
    for speaker in os.listdir(data_path):
        speaker_folder = os.path.join(data_path, speaker)
        if not os.path.isdir(speaker_folder):
            continue
        features = []
        for file in os.listdir(speaker_folder):
            if file.endswith(".wav"):
                file_path = os.path.join(speaker_folder, file)
                mfcc = extract_features(file_path)
                features.append(mfcc)
        features = np.vstack(features)
        gmm = GaussianMixture(n_components=4, covariance_type='diag', max_iter=100)
        gmm.fit(features)
        speaker_models[speaker] = gmm
    return speaker_models

def predict_speaker(models, test_file):
    mfcc = extract_features(test_file)
    scores = {}
    for speaker, model in models.items():
        score = model.score(mfcc)
        scores[speaker] = score
    predicted = max(scores, key=scores.get)
    return predicted
data_path = "dataset"
models = train_speaker_models(data_path)
test_file = "test.wav"
predicted = predict_speaker(models, test_file)
print(predicted)

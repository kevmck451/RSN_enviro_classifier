import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Load and extract features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

# Prepare dataset
def prepare_data(data_dir):
    features = []
    labels = []
    for label, env in enumerate(['car', 'bus', 'tram', 'train']):
        env_path = os.path.join(data_dir, env)
        for file_name in os.listdir(env_path):
            file_path = os.path.join(env_path, file_name)
            feature = extract_features(file_path)
            features.append(feature)
            labels.append(label)
    return np.array(features), np.array(labels)

# Load data
data_dir = '../data'
X, y = prepare_data(data_dir)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train classifier using Gaussian Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)

# Evaluate classifier
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['car', 'bus', 'tram', 'train']))

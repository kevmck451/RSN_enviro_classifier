import numpy as np
import librosa
import os
from sklearn.mixture import GaussianMixture
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


# Train Gaussian Mixture Models for each class
def train_gmm_models(X_train, y_train, num_classes, num_components=2):
    gmms = []
    for cls in range(num_classes):
        # Filter samples belonging to this class
        X_class = X_train[y_train == cls]

        # Train GMM
        gmm = GaussianMixture(n_components=num_components, covariance_type='full', random_state=42)
        gmm.fit(X_class)
        gmms.append(gmm)
    return gmms


# Predict class labels
def predict_classes(X_test, gmms):
    log_likelihoods = np.array([gmm.score_samples(X_test) for gmm in gmms]).T
    return np.argmax(log_likelihoods, axis=1)


# Main
if __name__ == "__main__":
    # Load data
    data_dir = '../data'
    X, y = prepare_data(data_dir)

    # Split into training and testing sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Train GMMs
    num_classes = len(np.unique(y))
    gmms = train_gmm_models(X_train, y_train, num_classes)

    # Predict on test set
    y_pred = predict_classes(X_test, gmms)

    # Evaluate
    print(classification_report(y_test, y_pred, target_names=['car', 'bus', 'tram', 'train']))

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data from JSON files
with open('GMM_summary.json', 'r') as file:
    gmm_data = json.load(file)

with open('GNB_summary.json', 'r') as file:
    gnb_data = json.load(file)


# Helper function to extract metrics
def extract_metrics(data, metric):
    metrics = {}
    for feature, details in data.items():
        if 'classification_report' in details:
            metrics[feature] = details['classification_report']['macro avg'][metric]
    return metrics


# Extract metrics for comparison
accuracy_gmm = {feature: details['accuracy'] for feature, details in gmm_data.items()}
accuracy_gnb = {feature: details['accuracy'] for feature, details in gnb_data.items()}
f1_gmm = extract_metrics(gmm_data, 'f1-score')
f1_gnb = extract_metrics(gnb_data, 'f1-score')

# Create accuracy comparison bar chart
features = list(accuracy_gmm.keys())
x = np.arange(len(features))

plt.figure(figsize=(10, 6))
plt.bar(x - 0.2, accuracy_gmm.values(), width=0.4, label='GMM', alpha=0.7)
plt.bar(x + 0.2, accuracy_gnb.values(), width=0.4, label='GNB', alpha=0.7, color='gray')
plt.xticks(x, features, rotation=45)
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison Across Features')
plt.legend()
plt.tight_layout()
plt.show()

# Create F1-score comparison bar chart
plt.figure(figsize=(10, 6))
plt.bar(x - 0.2, f1_gmm.values(), width=0.4, label='GMM', alpha=0.7)
plt.bar(x + 0.2, f1_gnb.values(), width=0.4, label='GNB', alpha=0.7)
plt.xticks(x, features, rotation=45)
plt.ylabel('Macro Avg F1-Score')
plt.title('F1-Score Comparison Across Features')
plt.legend()
plt.tight_layout()
plt.show()


# Create confusion matrix heatmaps
def plot_confusion_matrix(matrix, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap='Blues', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()


for feature in features:
    gmm_matrix = gmm_data[feature]['confusion_matrix']
    gnb_matrix = gnb_data[feature]['confusion_matrix']

    plot_confusion_matrix(gmm_matrix, f'GMM Confusion Matrix: {feature}')
    plt.show()

    plot_confusion_matrix(gnb_matrix, f'GNB Confusion Matrix: {feature}')
    plt.show()

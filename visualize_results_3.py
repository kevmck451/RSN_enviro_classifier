import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data from JSON files
with open('GMM_summary.json', 'r') as file:
    gmm_data = json.load(file)

with open('GNB_summary.json', 'r') as file:
    gnb_data = json.load(file)

# Helper functions to extract and consolidate data
def extract_metrics(data, metric):
    metrics = {}
    for feature, details in data.items():
        if 'classification_report' in details:
            metrics[feature] = details['classification_report']['macro avg'][metric]
    return metrics

def calculate_overall(data, metric):
    metrics = extract_metrics(data, metric)
    return pd.DataFrame({'Feature': list(metrics.keys()), metric: list(metrics.values())})

# Calculate overall metrics for accuracy, precision, recall, and F1-score
gmm_accuracy = pd.Series({feature: details['accuracy'] for feature, details in gmm_data.items()}, name="GMM_Accuracy")
gnb_accuracy = pd.Series({feature: details['accuracy'] for feature, details in gnb_data.items()}, name="GNB_Accuracy")

gmm_f1 = calculate_overall(gmm_data, 'f1-score')
gnb_f1 = calculate_overall(gnb_data, 'f1-score')

# Merge accuracy and F1-score data for comparison
accuracy_data = pd.concat([gmm_accuracy, gnb_accuracy], axis=1).reset_index().rename(columns={"index": "Feature"})
f1_data = pd.merge(gmm_f1, gnb_f1, on="Feature", suffixes=("_GMM", "_GNB"))

# Plot overall accuracy trends
plt.figure(figsize=(10, 6))
accuracy_data.set_index("Feature").plot(kind="bar", width=0.8, alpha=0.75, figsize=(10, 6))
plt.title("Overall Accuracy Comparison Across Features")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot macro F1-score trends
f1_data.set_index("Feature").plot(kind="bar", width=0.8, alpha=0.75, figsize=(10, 6))
plt.title("Macro-Averaged F1-Score Comparison Across Features")
plt.ylabel("F1-Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Consolidated heatmap of metrics
metrics_summary = pd.DataFrame({
    'Feature': accuracy_data['Feature'],
    'GMM_Accuracy': gmm_accuracy.values,
    'GNB_Accuracy': gnb_accuracy.values,
    'GMM_F1': gmm_f1['f1-score'].values,
    'GNB_F1': gnb_f1['f1-score'].values
}).set_index('Feature')

plt.figure(figsize=(10, 6))
plt.imshow(metrics_summary.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Performance Metric')
plt.title('Consolidated Performance Heatmap')
plt.xticks(ticks=np.arange(len(metrics_summary.index)), labels=metrics_summary.index, rotation=45)
plt.yticks(ticks=np.arange(len(metrics_summary.columns)), labels=metrics_summary.columns)
plt.tight_layout()
plt.show()

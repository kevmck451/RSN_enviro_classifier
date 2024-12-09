import json
import matplotlib.pyplot as plt
import pandas as pd

# Load data
with open('GMM_summary.json', 'r') as f:
    gmm = json.load(f)
with open('GNB_summary.json', 'r') as f:
    gnb = json.load(f)

# Extract metrics for all features
def extract_metrics(data, metric):
    result = []
    for feature, details in data.items():
        if 'classification_report' in details:
            value = details['classification_report']['macro avg'][metric]
            result.append({'Feature': feature, metric: value})
    return pd.DataFrame(result)

# Extract metrics for both models
gmm_metrics = extract_metrics(gmm, 'f1-score')
gmm_metrics['Model'] = 'GMM'
gnb_metrics = extract_metrics(gnb, 'f1-score')
gnb_metrics['Model'] = 'GNB'

# Combine data and calculate average metrics per feature
combined_metrics = pd.concat([gmm_metrics, gnb_metrics])
feature_performance = combined_metrics.groupby('Feature')['f1-score'].mean().sort_values(ascending=False)

# Plot 1: Average F1-score per feature (ranking)
plt.figure(figsize=(10, 6))
feature_performance.plot(kind='bar', color='skyblue', alpha=0.8)
plt.title("Average F1-Score by Feature (Across Models)")
plt.ylabel("Average F1-Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Overall model performance (average accuracy across features)
gmm_accuracy = pd.Series({k: v['accuracy'] for k, v in gmm.items()}, name="GMM_Accuracy")
gnb_accuracy = pd.Series({k: v['accuracy'] for k, v in gnb.items()}, name="GNB_Accuracy")
overall_accuracy = pd.DataFrame({'GMM': gmm_accuracy.mean(), 'GNB': gnb_accuracy.mean()}, index=["Model"]).T

plt.figure(figsize=(6, 4))
overall_accuracy.plot(kind='bar', legend=False, color=['blue', 'orange'], alpha=0.8)
plt.title("Overall Accuracy Comparison (Average Across Features)")
plt.ylabel("Accuracy")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Insights summary
top_features = feature_performance.head(3).index.tolist()
print(f"Top 3 Features Contributing to Model Performance: {top_features}")
print(f"Overall Accuracy:\n{overall_accuracy}")

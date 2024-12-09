import json
import pandas as pd
import matplotlib.pyplot as plt

# Load JSON files
with open('GMM_summary.json', 'r') as f:
    gmm_data = json.load(f)
with open('GNB_summary.json', 'r') as f:
    gnb_data = json.load(f)

# Extract metrics
def extract_performance(data, model_name):
    metrics = []
    for feature, details in data.items():
        metrics.append({
            'Feature': feature,
            'Model': model_name,
            'Accuracy': details['accuracy'],
            'F1-Score': details['classification_report']['macro avg']['f1-score'],
            'Precision': details['classification_report']['macro avg']['precision'],
            'Recall': details['classification_report']['macro avg']['recall']
        })
    return pd.DataFrame(metrics)

# Prepare DataFrames
gmm_metrics = extract_performance(gmm_data, 'GMM')
gnb_metrics = extract_performance(gnb_data, 'GNB')

# Combine DataFrames
metrics_df = pd.concat([gmm_metrics, gnb_metrics])

# Ensure numeric columns for aggregation
numeric_columns = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
metrics_df[numeric_columns] = metrics_df[numeric_columns].apply(pd.to_numeric)

# Calculate overall averages
model_comparison = metrics_df.groupby('Model')[numeric_columns].mean()

# Rank features by average F1-score across both models
feature_performance = metrics_df.groupby('Feature')[numeric_columns].mean().sort_values(by='F1-Score', ascending=False)

# Plot 1: Overall model performance
plt.figure(figsize=(8, 5))
model_comparison[['Accuracy', 'F1-Score']].plot(kind='bar', alpha=0.8, color=['blue', 'orange'])
plt.title('Overall Model Performance')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Plot 2: Feature ranking by F1-Score
plt.figure(figsize=(10, 6))
feature_performance['F1-Score'].plot(kind='bar', color='green', alpha=0.8)
plt.title('Feature Ranking by Average F1-Score')
plt.ylabel('F1-Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Insights
top_features = feature_performance.head(3).index.tolist()
better_model = model_comparison['F1-Score'].idxmax()

print("Key Insights:")
print(f"1. The top 3 features contributing to model performance are: {', '.join(top_features)}.")
print(f"2. The {better_model} model has better overall F1-Score.")

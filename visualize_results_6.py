import pandas as pd
import matplotlib.pyplot as plt

# Extract F1-scores for GMM and GNB
gmm_f1_scores = {
    "mfcc": [0.9487, 0.4727, 0.6250, 0.6087],
    "spectral_contrast": [0.9067, 0.6944, 0.9189, 0.7253],
    "combined": [0.7945, 0.6400, 0.2222, 0.5217]
}
gnb_f1_scores = {
    "mfcc": [0.8941, 0.6329, 0.5714, 0.6197],
    "spectral_contrast": [0.8421, 0.7727, 0.7838, 0.5946],
    "combined": [0.9176, 0.7901, 0.7297, 0.6944]
}

classes = ['car', 'bus', 'tram', 'train']


# DataFrames for visualization
df_gmm_f1 = pd.DataFrame(gmm_f1_scores, index=classes).T
df_gnb_f1 = pd.DataFrame(gnb_f1_scores, index=classes).T


# Plot
features = ['mfcc', 'spectral_contrast', 'combined']
classes = ['car', 'bus', 'tram', 'train']


fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# GMM
axes[0].plot(features, df_gmm_f1['car'], marker='o', color='blue', label='Car')
axes[0].plot(features, df_gmm_f1['bus'], marker='o', color='green', label='Bus')
axes[0].plot(features, df_gmm_f1['tram'], marker='o', color='gray', label='Tram')
axes[0].plot(features, df_gmm_f1['train'], marker='o', color='purple', label='Train')
axes[0].set_title("GMM F1-Score by Feature", fontsize=14)
axes[0].set_xlabel("Features", fontsize=12)
axes[0].set_ylabel("F1-Score", fontsize=12)
axes[0].legend(title="Class", fontsize=10)
axes[0].grid(True)

# GNB
axes[1].plot(features, df_gnb_f1['car'], marker='o', color='blue', label='Car')
axes[1].plot(features, df_gnb_f1['bus'], marker='o', color='green', label='Bus')
axes[1].plot(features, df_gnb_f1['tram'], marker='o', color='gray', label='Tram')
axes[1].plot(features, df_gnb_f1['train'], marker='o', color='purple', label='Train')
axes[1].set_title("GNB F1-Score by Feature", fontsize=14)
axes[1].set_xlabel("Features", fontsize=12)
axes[1].legend(title="Class", fontsize=10)
axes[1].grid(True)

plt.tight_layout()
plt.show()

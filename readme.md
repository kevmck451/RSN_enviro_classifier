# Random Signals and Noise Project

## Environmental Audio Classifier
- Car
- Bus
- Tram
- Train

## Features Investigated

- Temporal Features vs Spectral Features


- mean frequency
  - Freq vs Magnitude
  - Freq vs Magnitude (Normalized)
  - Freq vs Magnitude (dB Scale)
  - Freq vs Magnitude (dB Scale and Normalized)
- power spectral density
- mfcc
  - top 13 coefficients
  - magnitudes mean
- spectral contrast
  - freq band vs mean contrast (dB)
- spectral centroid
  - time vs freq
- chroma features
  - mean chroma vectors vs energy
- rms energy
  - mean rms over time
- zero crossing rate
  - mean
  - variance


## Models

- GMM: gaussian mixture model
- Naive Bayes Classifier




# Outline for Comparing GMM and Naive Bayes Classifier Performance

## Feature Extraction
- Extract multiple types of features from audio samples:
  - **MFCCs**: Standard feature set for audio classification.
  - **Spectral Features**: Centroid, bandwidth, contrast, roll-off.
  - **Temporal Features**: Energy, RMS, zero-crossing rate.
  - **Statistical Features**: Mean, variance, skewness, kurtosis of the raw signal or derived features.

### Data Normalization
- Standardize features to have zero mean and unit variance for comparability.

### Feature Engineering
- Concatenate features into a unified feature vector for each sample.
- Optionally, test subsets of features to determine their impact on model performance.

## Model Training
### Gaussian Mixture Model
- Train separate GMMs for each class (car, bus, tram, train).
- Use Expectation-Maximization (EM) to estimate parameters (mean, covariance, weights).
- Experiment with the number of Gaussian components to optimize performance.

### Naive Bayes Classifier
- Assume features are independent given the class.
- Estimate feature likelihoods using Gaussian distributions or other probability distributions (e.g., multinomial, if features are categorical).
- Train one model for all classes.

## Evaluation Pipeline
### Data Splitting
- Divide the dataset into training (70%) and testing (30%) sets.
- Use stratified sampling to ensure balanced representation of classes.

### Cross-Validation
- Perform k-fold cross-validation (e.g., k=5) to assess generalization performance.

### Metrics
- **Accuracy**: Proportion of correctly classified samples.
- **Confusion Matrix**: Assess per-class performance.
- **Precision, Recall, F1-Score**: For each class.
- **Log-Likelihood** (GMM only): Measure how well the model fits the data.

### Statistical Comparison
- Use paired statistical tests (e.g., paired t-test) to determine if differences in performance between GMM and NBC are statistically significant.

## Feature Impact Analysis
### Per-Feature Analysis
- Train and test models with individual feature types to evaluate their contribution to performance.

### Feature Subset Evaluation
- Use subsets of features (e.g., MFCCs + spectral features) to compare performance and identify synergistic feature combinations.

### Feature Importance
- For Naive Bayes, compare likelihood ratios to determine feature importance.
- For GMM, evaluate the impact of each feature on the log-likelihood of the model.

## Experiment Logging and Visualization
### Log Results
- Record training/testing accuracy, confusion matrices, and evaluation metrics for each model and feature set.

### Visualization
- Compare metrics using bar charts (e.g., accuracy, F1-score).
- Plot confusion matrices for each model.
- Visualize the likelihood distribution for each class in GMM.

## Conclusion and Recommendations
### Model Comparison
- Summarize which model performs better overall and for specific classes.
- Identify any trade-offs (e.g., GMM may handle feature correlations better, NBC may be faster).

### Feature Insights
- Highlight the most discriminative features or combinations.

### Recommendations
- Suggest which model is preferable for deployment based on accuracy, computation time, and robustness.

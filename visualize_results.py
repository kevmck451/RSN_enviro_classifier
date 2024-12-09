import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load results from JSON file
def load_results(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


# Plot accuracy comparison
def plot_accuracy_comparison(results):
    feature_types = []
    accuracies = []

    for feature, metrics in results.items():
        feature_types.append(feature)
        accuracies.append(metrics["accuracy"])

    plt.figure(figsize=(10, 6))
    plt.bar(feature_types, accuracies, color="skyblue")
    plt.title("Accuracy Comparison Across Feature Types")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Plot precision, recall, and F1-score comparison for each class
def plot_classwise_metrics(results):
    metrics = ["precision", "recall", "f1-score"]
    classes = ["car", "bus", "tram", "train"]

    for metric in metrics:
        plt.figure(figsize=(12, 8))
        for feature, data in results.items():
            scores = [data["classification_report"][cls][metric] for cls in classes]
            plt.plot(classes, scores, marker="o", label=feature)

        plt.title(f"Class-wise {metric.capitalize()} Comparison Across Feature Types")
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


# Plot confusion matrix for each feature type
def plot_confusion_matrices(results):
    for feature, metrics in results.items():
        cm = np.array(metrics["confusion_matrix"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["car", "bus", "tram", "train"], yticklabels=["car", "bus", "tram", "train"])
        plt.title(f"Confusion Matrix for {feature}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()


# Main
if __name__ == "__main__":
    json_file = "GMM_summary.json"  # Replace with your JSON file path
    results = load_results(json_file)

    # Generate visualizations
    plot_accuracy_comparison(results)
    plot_classwise_metrics(results)
    plot_confusion_matrices(results)

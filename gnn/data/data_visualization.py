import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json
import os
from sklearn.metrics import roc_curve, auc
# ChatGPT
def plot_roc_curve(y_true, y_scores, dataset_name="ROC", save_path="visualizations"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{dataset_name} ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{dataset_name}_roc_curve.png"))
    plt.close()
    print(f"Saved {dataset_name} ROC curve to {save_path}")


def plot_loss(losses, path="loss_curve.png"):
    epoch_losses = losses.get("epoch_loss", [])
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.title("Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved loss curve to {path}")


def plot_confusion_matrix(y_true, y_pred, dataset_name, save_path="visualizations"):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Safe", "Vulnerable"], yticklabels=["Safe", "Vulnerable"])
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{dataset_name}_conf_matrix.png"))
    plt.close()
    print(f"Saved {dataset_name.lower()} confusion matrix to {save_path}")

def plot_training_history(history_file_path, save_dir):
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    with open(history_file_path, "r") as f:
        history = json.load(f)

    epochs = history["epoch"]

    # === Plot 1: Training Loss ===
    plt.figure()
    plt.plot(epochs, history["train_loss"], marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "train_loss_curve.png"))
    plt.close()

    # === Plot 2: Training vs Validation Accuracy ===
    plt.figure()
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy", marker='o')
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy", marker='x')
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()

    # === Plot 3: Validation Precision, Recall, F1 ===
    plt.figure()
    plt.plot(epochs, history["val_precision"], label="Precision", marker='o')
    plt.plot(epochs, history["val_recall"], label="Recall", marker='x')
    plt.plot(epochs, history["val_f1"], label="F1 Score", marker='s')
    plt.title("Validation Metrics per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "val_metrics_curve.png"))
    plt.close()

    print("✅ Training history plots saved to:", save_dir)

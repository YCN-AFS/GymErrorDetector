import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import torch

# Hàm vẽ biểu đồ loss và accuracy trong quá trình training
def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Model Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Model Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Hàm vẽ confusion matrix
def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

# Hàm vẽ ROC curve
def plot_roc_curve(y_true, y_scores, save_path='roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Hàm phân tích phân phối đặc trưng
def plot_feature_distributions(data, save_path='feature_distributions.png'):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(data.columns[:12], 1):  # Plot first 12 features
        plt.subplot(3, 4, i)
        sns.histplot(data=data, x=col, hue='class', bins=30)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Hàm vẽ correlation matrix
def plot_correlation_matrix(data, save_path='correlation_matrix.png'):
    plt.figure(figsize=(12, 10))
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Thêm hàm vẽ Precision-Recall curve
def plot_precision_recall_curve(y_true, y_scores, save_path='pr_curve.png'):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Thêm hàm vẽ Learning Rate over time
def plot_learning_rate(learning_rates, save_path='learning_rate.png'):
    plt.figure(figsize=(8, 6))
    plt.plot(learning_rates)
    plt.title('Learning Rate over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Load data
    data = pd.read_csv("video_poses.csv")
    
    # Vẽ phân phối đặc trưng
    plot_feature_distributions(data)
    
    # Vẽ correlation matrix
    plot_correlation_matrix(data)
    
    # Giả sử bạn có history từ quá trình training
    # plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Giả sử bạn có predictions từ model
    # plot_confusion_matrix(y_true, y_pred)
    # plot_roc_curve(y_true, y_scores) 
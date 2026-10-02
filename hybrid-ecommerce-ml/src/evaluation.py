import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)

def evaluate_model_performance(y_true, y_pred, y_prob):
    """Calculates comprehensive metric summary for purchase conversion models."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        'PR-AUC': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        'Brier Score': brier_score_loss(y_true, y_prob)
    }
    return metrics

def plot_confusion_matrix_fig(y_true, y_pred, save_path=None):
    """Generates confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_xticklabels(["No Purchase (0)", "Purchase (1)"])
    ax.set_yticklabels(["No Purchase (0)", "Purchase (1)"])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig

def plot_roc_curve_fig(y_true, y_prob, save_path=None):
    """Generates ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_val:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig

def plot_feature_importance_fig(model, feature_names, top_n=15, save_path=None):
    """Generates feature importance visualization highlighting engineered features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]
    
    colors = ['#ff7f0e' if name in ['cluster_id', 'is_outlier', 'similarity_score'] else '#1f77b4' for name in top_names]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(top_names)), top_vals[::-1], color=colors[::-1], align='center')
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Relative Importance")
    ax.set_title(f"Top {top_n} Feature Importances (Engineered Features Highlighted)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig
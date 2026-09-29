"""
Evaluation and Visualization Module for Home Credit Default Risk.

Generates:
- Validation ROC-AUC, classification report, confusion matrix
- ROC Curve plot
- Confusion Matrix plot
- Feature Importance CSV and visualization
"""

import os
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)

from src.utils import get_logger

logger = get_logger("evaluate")


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics for binary classification.
    """
    auc = roc_auc_score(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    # Binary predictions based on threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, output_dict=False)
    
    logger.info(f"==================================================")
    logger.info(f"EVALUATION RESULTS: {model_name}")
    logger.info(f"==================================================")
    logger.info(f"ROC-AUC Score: {auc:.5f}")
    logger.info(f"PR-AUC (Average Precision): {avg_precision:.5f}")
    logger.info(f"Confusion Matrix (Threshold={threshold}):\n{cm}")
    logger.info(f"Classification Report:\n{cr}")
    logger.info(f"==================================================")
    
    return {
        "roc_auc": auc,
        "pr_auc": avg_precision,
        "confusion_matrix": cm,
        "classification_report": cr,
        "y_pred_proba": y_pred_proba
    }


def plot_roc_curve(
    models_dict: Dict[str, Dict[str, Any]],
    save_path: str = "reports/figures/roc_curve.png"
):
    """
    Plot ROC curves for multiple models on the same figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess (AUC = 0.5000)')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (name, data) in enumerate(models_dict.items()):
        y_true = data['y_true']
        y_proba = data['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        color = colors[idx % len(colors)]
        plt.plot(fpr, tpr, lw=2.2, label=f"{name} (AUC = {auc:.4f})", color=color)
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curve plot to {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str = "Final Model",
    save_path: str = "reports/figures/confusion_matrix.png"
):
    """
    Plot formatted confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        annot_kws={'size': 14, 'weight': 'bold'},
        xticklabels=['Non-Default (0)', 'Default (1)'],
        yticklabels=['Non-Default (0)', 'Default (1)']
    )
    plt.title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {save_path}")


def save_and_plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 30,
    csv_path: str = "reports/feature_importance.csv",
    fig_path: str = "reports/figures/feature_importance.png"
) -> pd.DataFrame:
    """
    Extract, sort, save to CSV, and plot top N feature importances.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    # Save complete importance table
    fi_df.to_csv(csv_path, index=False)
    logger.info(f"Saved complete feature importances ({len(fi_df)} features) to {csv_path}")
    
    # Plot top N
    top_fi = fi_df.head(top_n).copy()
    plt.figure(figsize=(12, 10))
    sns.barplot(
        x='importance',
        y='feature',
        data=top_fi,
        palette='viridis'
    )
    plt.title(f'Top {top_n} Most Important Features (LightGBM Gain/Split)', fontsize=15, fontweight='bold', pad=14)
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Feature Name', fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved top {top_n} feature importance plot to {fig_path}")
    
    return fi_df

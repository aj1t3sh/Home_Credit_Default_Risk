"""
Model Training and Selection Module for Home Credit Default Risk.

Trains:
1. Baseline Logistic Regression (with Imputation + Scaling)
2. LightGBM Gradient Boosted Decision Trees (with early stopping)

Compares models using validation ROC-AUC, selects the top performer,
retrains on the full dataset, and persists models/final_model.pkl.
"""

import gc
import joblib
import logging
import os
from typing import Dict, Any, Tuple, List
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import (
    evaluate_predictions,
    plot_roc_curve,
    plot_confusion_matrix,
    save_and_plot_feature_importance
)
from src.utils import timer, get_logger

logger = get_logger("train")


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a baseline Logistic Regression pipeline with median imputation and standardization.
    """
    with timer("Training Baseline Logistic Regression", logger):
        lr_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=0.01,
                max_iter=250,
                class_weight='balanced',
                solver='lbfgs',
                random_state=random_state,
                n_jobs=-1
            ))
        ])
        
        logger.info("Fitting Logistic Regression baseline...")
        lr_pipeline.fit(X_train, y_train)
        
        y_val_proba = lr_pipeline.predict_proba(X_val)[:, 1]
        results = evaluate_predictions(y_val.values, y_val_proba, model_name="Logistic Regression Baseline")
        results['y_true'] = y_val.values
        
        return lr_pipeline, results


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42
) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
    """
    Train a gradient-boosted decision tree classifier using LightGBM with early stopping.
    """
    with timer("Training LightGBM Gradient Boosting Model", logger):
        pos_count = int(y_train.sum())
        neg_count = int(len(y_train) - pos_count)
        scale_pos_weight = neg_count / max(1, pos_count)
        logger.info(f"Class imbalance: {neg_count:,} negative vs {pos_count:,} positive (scale_pos_weight={scale_pos_weight:.2f})")
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            objective='binary',
            metric='auc',
            random_state=random_state,
            n_jobs=-1,
            importance_type='gain',
            verbose=-1
        )
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
        
        logger.info("Fitting LightGBM model with early stopping on validation ROC-AUC...")
        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=['train', 'val'],
            callbacks=callbacks
        )
        
        y_val_proba = lgb_model.predict_proba(X_val)[:, 1]
        results = evaluate_predictions(y_val.values, y_val_proba, model_name="LightGBM Classifier")
        results['y_true'] = y_val.values
        results['best_iteration'] = lgb_model.best_iteration_
        
        return lgb_model, results


def run_training_pipeline(
    train_df: pd.DataFrame,
    project_root: str = None,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any], List[str]]:
    """
    Orchestrates the model comparison, validation, selection, feature importance generation,
    and full dataset retraining.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    features = [c for c in train_df.columns if c not in ['TARGET', 'SK_ID_CURR']]
    X = train_df[features]
    y = train_df['TARGET']
    
    logger.info(f"Dataset summary: Total samples = {len(X):,}, Total features = {len(features)}")
    logger.info(f"Target distribution: 0 = {(y == 0).sum():,} ({(y == 0).mean()*100:.2f}%), 1 = {(y == 1).sum():,} ({(y == 1).mean()*100:.2f}%)")
    
    # Stratified Train/Val split
    logger.info("Creating 80/20 stratified train/validation split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )
    logger.info(f"Train split shape: {X_train.shape}, Validation split shape: {X_val.shape}")
    
    # 1. Train Baseline
    lr_model, lr_results = train_logistic_regression(X_train, y_train, X_val, y_val, random_state)
    
    # 2. Train LightGBM
    lgb_model, lgb_results = train_lightgbm(X_train, y_train, X_val, y_val, random_state)
    
    # 3. Model Comparison
    logger.info("==================================================")
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("==================================================")
    logger.info(f"1. Logistic Regression Baseline Val ROC-AUC: {lr_results['roc_auc']:.5f}")
    logger.info(f"2. LightGBM Gradient Boosting Val ROC-AUC:   {lgb_results['roc_auc']:.5f}")
    logger.info("==================================================")
    
    # Plot ROC Comparison
    models_dict = {
        "Logistic Regression Baseline": lr_results,
        "LightGBM Gradient Boosting": lgb_results
    }
    plot_roc_curve(models_dict, save_path=os.path.join(project_root, "reports", "figures", "roc_curve.png"))
    
    # Feature Importance for LightGBM
    importances = lgb_model.feature_importances_
    save_and_plot_feature_importance(
        feature_names=features,
        importances=importances,
        top_n=30,
        csv_path=os.path.join(project_root, "reports", "feature_importance.csv"),
        fig_path=os.path.join(project_root, "reports", "figures", "feature_importance.png")
    )
    
    # Confusion Matrix for LightGBM
    plot_confusion_matrix(
        lgb_results['confusion_matrix'],
        model_name="LightGBM Classifier",
        save_path=os.path.join(project_root, "reports", "figures", "confusion_matrix.png")
    )
    
    # 4. Final Retraining on Full Dataset
    best_iteration = lgb_results['best_iteration']
    n_estimators_full = int(best_iteration * 1.15) if best_iteration else 1000
    logger.info(f"Retraining final LightGBM model on full training set ({len(X):,} rows) with n_estimators={n_estimators_full}...")
    
    final_model = lgb.LGBMClassifier(
        n_estimators=n_estimators_full,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective='binary',
        metric='auc',
        random_state=random_state,
        n_jobs=-1,
        importance_type='gain',
        verbose=-1
    )
    
    with timer("Full Training Dataset Fit", logger):
        final_model.fit(X, y)
        
    # Save final model
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "final_model.pkl")
    joblib.dump(final_model, model_save_path)
    logger.info(f"Persisted final model artifact to {model_save_path}")
    
    eval_summary = {
        "lr_val_auc": lr_results['roc_auc'],
        "lgb_val_auc": lgb_results['roc_auc'],
        "best_model_name": "LightGBM Classifier",
        "best_iteration": best_iteration,
        "n_train_rows": len(X),
        "n_features": len(features)
    }
    
    return final_model, eval_summary, features

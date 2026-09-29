#!/usr/bin/env python3
"""
Home Credit Default Risk - End-to-End Production ML Pipeline.

Executes:
1. Exploratory Data Analysis & Visualization generation
2. Base Application Data Preprocessing & Categorical Alignment
3. Relational Table Aggregations & Domain Feature Engineering
4. Stratified Train/Val Split & Model Comparison (Logistic Regression vs LightGBM)
5. Model Evaluation (ROC-AUC, Confusion Matrix, Feature Importance)
6. Full Retraining & Model Persistence (models/final_model.pkl)
7. Test Inference & Kaggle submission.csv generation and verification
"""

import gc
import os
import sys
import pandas as pd

from src.data_preprocessing import preprocess_application_data
from src.eda_analysis import run_eda
from src.feature_engineering import build_full_feature_matrix
from src.predict import generate_submission
from src.train import run_training_pipeline
from src.utils import get_data_path, get_logger, timer

logger = get_logger("run_pipeline")


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    logger.info("================================================================")
    logger.info("STARTING HOME CREDIT DEFAULT RISK MACHINE LEARNING PIPELINE")
    logger.info("================================================================")
    
    # 1. Exploratory Data Analysis
    logger.info("STEP 1/6: Running Exploratory Data Analysis...")
    eda_summary = run_eda(project_root)
    
    # 2. Data Preprocessing
    logger.info("STEP 2/6: Preprocessing Application Datasets...")
    train_app, test_app = preprocess_application_data(project_root)
    
    # 3. Feature Engineering & Relational Merges
    logger.info("STEP 3/6: Building Full Feature Matrix from Relational Datasets...")
    processed_train_path = os.path.join(project_root, "data", "processed", "train_features.parquet")
    processed_test_path = os.path.join(project_root, "data", "processed", "test_features.parquet")
    
    # Cache processed tables to allow quick iteration
    if os.path.exists(processed_train_path) and os.path.exists(processed_test_path):
        logger.info(f"Loading cached feature matrices from {processed_train_path}...")
        train_df = pd.read_parquet(processed_train_path)
        test_df = pd.read_parquet(processed_test_path)
    else:
        train_df, test_df = build_full_feature_matrix(train_app, test_app, project_root)
        os.makedirs(os.path.join(project_root, "data", "processed"), exist_ok=True)
        logger.info(f"Saving engineered feature sets to {processed_train_path} and {processed_test_path}...")
        train_df.to_parquet(processed_train_path, index=False)
        test_df.to_parquet(processed_test_path, index=False)
        
    del train_app, test_app
    gc.collect()
    
    # 4. Model Training, Validation, and Selection
    logger.info("STEP 4/6: Training, Evaluating, and Selecting Models...")
    final_model, eval_summary, feature_names = run_training_pipeline(train_df, project_root)
    
    # 5. Prediction and Kaggle Submission
    logger.info("STEP 5/6: Generating and Validating Test Predictions...")
    sub_path = os.path.join(project_root, "submission.csv")
    sample_sub_path = None
    try:
        sample_sub_path = get_data_path("sample_submission.csv", project_root)
    except FileNotFoundError:
        pass
        
    submission_df = generate_submission(
        model=final_model,
        test_df=test_df,
        features=feature_names,
        output_path=sub_path,
        sample_sub_path=sample_sub_path
    )
    
    # 6. Final Report & Verification
    logger.info("STEP 6/6: Execution Completed Successfully!")
    print("\n" + "="*70)
    print("                 PIPELINE EXECUTION SUMMARY")
    print("="*70)
    print(f"Training rows:               {eval_summary['n_train_rows']:,}")
    print(f"Test rows:                   {len(test_df):,}")
    print(f"Number of final features:    {eval_summary['n_features']:,}")
    print(f"Logistic Regression Val AUC: {eval_summary['lr_val_auc']:.5f}")
    print(f"LightGBM Val ROC-AUC:        {eval_summary['lgb_val_auc']:.5f}")
    print(f"Selected model:              {eval_summary['best_model_name']}")
    print(f"Submission file path:        {sub_path}")
    print(f"Submission shape:            {submission_df.shape}")
    print(f"Submission columns:          {list(submission_df.columns)}")
    print("\nSubmission Preview (First 10 rows):")
    print(submission_df.head(10).to_string(index=False))
    print("\nSubmission Probability Distribution Summary:")
    print(submission_df['TARGET'].describe().to_string())
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

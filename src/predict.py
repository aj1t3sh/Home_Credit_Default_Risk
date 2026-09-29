"""
Inference and Kaggle Submission Generation Module for Home Credit Default Risk.

Generates submission.csv, performs exhaustive validation checks against sample_submission.csv,
and verifies probabilities are well-behaved.
"""

import os
from typing import List, Optional
import numpy as np
import pandas as pd

from src.utils import get_data_path, timer, get_logger

logger = get_logger("predict")


def generate_submission(
    model,
    test_df: pd.DataFrame,
    features: List[str],
    output_path: str = "submission.csv",
    sample_sub_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate test set probability predictions and format into a verified Kaggle submission.csv.
    """
    with timer("Generating Test Predictions and Submission File", logger):
        if 'SK_ID_CURR' not in test_df.columns:
            raise KeyError("test_df must contain 'SK_ID_CURR' column.")
            
        test_ids = test_df['SK_ID_CURR'].values
        X_test = test_df[features]
        
        logger.info(f"Computing test predictions for {len(X_test):,} applicants with {len(features)} features...")
        test_probs = model.predict_proba(X_test)[:, 1]
        
        submission = pd.DataFrame({
            'SK_ID_CURR': test_ids.astype(int),
            'TARGET': test_probs.astype(float)
        })
        
        # Validation Checks
        logger.info("Executing rigorous submission validation checks...")
        
        # 1. Shape check
        assert submission.shape[0] == len(test_df), f"Row count mismatch: {submission.shape[0]} vs expected {len(test_df)}"
        assert submission.shape[1] == 2, f"Column count mismatch: {submission.shape[1]} vs expected 2"
        
        # 2. Columns check
        expected_cols = ['SK_ID_CURR', 'TARGET']
        assert list(submission.columns) == expected_cols, f"Columns mismatch: {list(submission.columns)} vs {expected_cols}"
        
        # 3. Missing values check
        assert submission['SK_ID_CURR'].isnull().sum() == 0, "Null values found in SK_ID_CURR!"
        assert submission['TARGET'].isnull().sum() == 0, "Null values found in TARGET!"
        
        # 4. Duplicate IDs check
        assert submission['SK_ID_CURR'].duplicated().sum() == 0, "Duplicate SK_ID_CURR values found!"
        
        # 5. Probability range check
        min_p = submission['TARGET'].min()
        max_p = submission['TARGET'].max()
        mean_p = submission['TARGET'].mean()
        assert 0.0 <= min_p <= 1.0, f"Min probability out of bounds: {min_p}"
        assert 0.0 <= max_p <= 1.0, f"Max probability out of bounds: {max_p}"
        
        # 6. Continuous variance check (not binary 0/1)
        unique_probs = submission['TARGET'].nunique()
        assert unique_probs > 1000, f"Predicted probabilities appear non-continuous (unique count = {unique_probs})"
        
        # 7. Check against sample_submission if available
        if sample_sub_path and os.path.exists(sample_sub_path):
            sample_sub = pd.read_csv(sample_sub_path)
            assert len(submission) == len(sample_sub), f"Row count doesn't match sample_submission: {len(submission)} vs {len(sample_sub)}"
            assert (submission['SK_ID_CURR'].values == sample_sub['SK_ID_CURR'].values).all(), "SK_ID_CURR ordering does not match sample_submission!"
            logger.info("Sample submission alignment verification: PASSED (IDs and ordering match perfectly).")
            
        # Save to CSV
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        submission.to_csv(output_path, index=False)
        logger.info(f"Successfully saved Kaggle submission to: {output_path}")
        logger.info(f"Submission shape: {submission.shape}")
        logger.info(f"Target probability statistics: min={min_p:.5f}, max={max_p:.5f}, mean={mean_p:.5f}")
        
    return submission

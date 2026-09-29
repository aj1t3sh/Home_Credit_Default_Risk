"""
Data Preprocessing Module for Home Credit Default Risk.

Handles:
- Loading application_train.csv and application_test.csv
- Handling anomalies (e.g. DAYS_EMPLOYED == 365243)
- Encoding categorical variables (Label Encoding for binary, One-Hot Encoding for multi-class)
- Aligning training and test feature spaces
"""

import gc
import logging
import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils import get_data_path, reduce_mem_usage, timer, get_logger

logger = get_logger("data_preprocessing")


def clean_application_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle known domain anomalies in application data:
    - DAYS_EMPLOYED: 365243 is a known placeholder for positive infinity / retired / missing.
      Replace with NaN and create indicator feature DAYS_EMPLOYED_ANOM.
    - CODE_GENDER: Replace 'XNA' (4 records) with NaN.
    - ORGANIZATION_TYPE: Replace 'XNA' with NaN.
    """
    df = df.copy()
    
    # Days employed anomaly
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(np.int8)
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})
        
    # Code gender anomaly
    if 'CODE_GENDER' in df.columns:
        df['CODE_GENDER'] = df['CODE_GENDER'].replace({'XNA': np.nan})
        
    # Organization type anomaly
    if 'ORGANIZATION_TYPE' in df.columns:
        df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].replace({'XNA': np.nan})
        
    return df


def preprocess_application_data(
    project_root: Optional[str] = None,
    nan_as_category: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and preprocesses application_train and application_test datasets.
    
    Returns:
        train_df: Preprocessed training DataFrame containing SK_ID_CURR and TARGET
        test_df: Preprocessed test DataFrame containing SK_ID_CURR
    """
    with timer("Loading and Preprocessing Application Data", logger):
        train_path = get_data_path("application_train.csv", project_root)
        test_path = get_data_path("application_test.csv", project_root)
        
        logger.info(f"Reading {train_path} and {test_path}...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Raw train shape: {train_df.shape}, raw test shape: {test_df.shape}")
        
        # Clean anomalies
        train_df = clean_application_anomalies(train_df)
        test_df = clean_application_anomalies(test_df)
        
        # Identify categorical columns
        cat_cols = [
            col for col in train_df.columns 
            if col != 'TARGET' and (train_df[col].dtype == 'object' or pd.api.types.is_string_dtype(train_df[col]))
        ]
        logger.info(f"Identified {len(cat_cols)} categorical columns: {cat_cols}")
        
        # Categorical Encoding:
        # 1. Binary features (<= 2 unique values) -> Label Encode
        # 2. Multi-class features (> 2 unique values) -> One-Hot Encode
        binary_cols = []
        multi_cols = []
        
        for col in cat_cols:
            train_vals = set(train_df[col].dropna().unique())
            test_vals = set(test_df[col].dropna().unique()) if col in test_df.columns else set()
            combined_unique = train_vals.union(test_vals)
            
            if len(combined_unique) <= 2:
                binary_cols.append(col)
                # Map values safely
                mapping = {val: idx for idx, val in enumerate(sorted(list(combined_unique)))}
                train_df[col] = train_df[col].map(mapping).astype(float)
                if col in test_df.columns:
                    test_df[col] = test_df[col].map(mapping).astype(float)
            else:
                multi_cols.append(col)
                
        logger.info(f"Label encoded {len(binary_cols)} binary categorical columns: {binary_cols}")
        logger.info(f"One-hot encoding {len(multi_cols)} multi-categorical columns: {multi_cols}")
        
        train_df = pd.get_dummies(train_df, columns=multi_cols, dummy_na=nan_as_category)
        test_df = pd.get_dummies(test_df, columns=multi_cols, dummy_na=nan_as_category)
        
        # Align columns between train and test
        train_cols = [c for c in train_df.columns if c not in ['TARGET', 'SK_ID_CURR']]
        test_cols = [c for c in test_df.columns if c not in ['SK_ID_CURR']]
        
        common_cols = sorted(list(set(train_cols).intersection(set(test_cols))))
        logger.info(f"Common features after alignment: {len(common_cols)}")
        
        train_df = train_df[['SK_ID_CURR'] + common_cols + ['TARGET']].copy()
        test_df = test_df[['SK_ID_CURR'] + common_cols].copy()
        
        train_df = reduce_mem_usage(train_df, verbose=False, logger=logger)
        test_df = reduce_mem_usage(test_df, verbose=False, logger=logger)
        
        logger.info(f"Preprocessed train shape: {train_df.shape}, test shape: {test_df.shape}")
        
    return train_df, test_df

"""
Feature Engineering Module for Home Credit Default Risk.

Builds domain features on the application dataset and aggregates secondary relational datasets:
- bureau.csv and bureau_balance.csv
- previous_application.csv
- POS_CASH_balance.csv
- installments_payments.csv
- credit_card_balance.csv
"""

import gc
import logging
import os
import re
from typing import Tuple, Optional
import numpy as np
import pandas as pd

from src.utils import get_data_path, reduce_mem_usage, timer, get_logger

logger = get_logger("feature_engineering")


def is_text_column(series: pd.Series) -> bool:
    """Check if a column contains text/object/string data."""
    return series.dtype == 'object' or pd.api.types.is_string_dtype(series)


def build_application_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct high-signal domain ratios and external source statistics for application data.
    """
    df = df.copy()
    
    # Financial Ratios
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / (df['AMT_CREDIT'] + 1e-5)
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1e-5)
    df['CREDIT_GOODS_PERC'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1e-5)
    
    # Demographic / Household Ratios
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'].fillna(1) + 1e-5)
    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    
    # Employment and Age Ratios
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1e-5)
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25
    df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED'] / 365.25
    
    if 'OWN_CAR_AGE' in df.columns:
        df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / (df['AGE_YEARS'] + 1e-5)
        df['CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / (df['EMPLOYED_YEARS'] + 1e-5)
        
    # External Sources Interactions
    ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df.columns]
    if len(ext_cols) == 3:
        df['EXT_SOURCES_MEAN'] = df[ext_cols].mean(axis=1)
        df['EXT_SOURCES_STD'] = df[ext_cols].std(axis=1)
        df['EXT_SOURCES_MIN'] = df[ext_cols].min(axis=1)
        df['EXT_SOURCES_MAX'] = df[ext_cols].max(axis=1)
        df['EXT_SOURCES_SUM'] = df[ext_cols].sum(axis=1)
        df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        
        # Cross Ratios
        df['EXT_SOURCE_1_2_RATIO'] = df['EXT_SOURCE_1'] / (df['EXT_SOURCE_2'] + 1e-5)
        df['EXT_SOURCE_2_3_RATIO'] = df['EXT_SOURCE_2'] / (df['EXT_SOURCE_3'] + 1e-5)
        df['EXT_SOURCE_1_3_RATIO'] = df['EXT_SOURCE_1'] / (df['EXT_SOURCE_3'] + 1e-5)
        
    return df


def aggregate_bureau_and_balance(project_root: Optional[str] = None, nan_as_category: bool = True) -> pd.DataFrame:
    """
    Process and aggregate bureau.csv and bureau_balance.csv by SK_ID_CURR.
    """
    with timer("Bureau and Bureau Balance Aggregations", logger):
        bureau_path = get_data_path("bureau.csv", project_root)
        bb_path = get_data_path("bureau_balance.csv", project_root)
        
        # 1. Process bureau_balance.csv
        logger.info("Reading bureau_balance.csv...")
        bb = pd.read_csv(bb_path)
        bb = reduce_mem_usage(bb, verbose=False, logger=logger)
        
        bb_cat_cols = [c for c in bb.columns if is_text_column(bb[c])]
        bb = pd.get_dummies(bb, columns=bb_cat_cols, dummy_na=nan_as_category, dtype=float)
        
        bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
        for col in bb.columns:
            if col not in ['SK_ID_BUREAU', 'MONTHS_BALANCE'] and pd.api.types.is_numeric_dtype(bb[col]):
                bb_aggregations[col] = ['mean', 'sum']
                
        bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
        bb_agg.columns = pd.Index([f"BB_{e[0]}_{e[1].upper()}" for e in bb_agg.columns.tolist()])
        
        del bb
        gc.collect()
        
        # 2. Process bureau.csv
        logger.info("Reading bureau.csv...")
        bureau = pd.read_csv(bureau_path)
        bureau = reduce_mem_usage(bureau, verbose=False, logger=logger)
        
        # Merge bureau_balance aggregated features
        bureau = bureau.merge(bb_agg, how='left', on='SK_ID_BUREAU')
        del bb_agg
        gc.collect()
        
        # Categorical encoding for bureau
        cat_cols = [c for c in bureau.columns if is_text_column(bureau[c])]
        bureau = pd.get_dummies(bureau, columns=cat_cols, dummy_na=nan_as_category, dtype=float)
        
        # Bureau numeric aggregations
        num_aggregations = {
            'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean', 'sum'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_ANNUITY': ['max', 'mean'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'MONTHS_BALANCE_min' if 'MONTHS_BALANCE_min' in bureau.columns else 'SK_ID_BUREAU': ['size']
        }
        
        # Filter and add dummy and BB numeric columns
        for c in bureau.columns:
            if (c.startswith('CREDIT_') or c.startswith('BB_')) and c not in num_aggregations and c != 'SK_ID_CURR' and c != 'SK_ID_BUREAU':
                if pd.api.types.is_numeric_dtype(bureau[c]):
                    num_aggregations[c] = ['mean']
                    
        # Filter num_aggregations to only valid existing columns
        valid_aggs = {k: v for k, v in num_aggregations.items() if k in bureau.columns and pd.api.types.is_numeric_dtype(bureau[k])}
        
        # Group by SK_ID_CURR
        bureau_agg = bureau.groupby('SK_ID_CURR').agg(valid_aggs)
        bureau_agg.columns = pd.Index([f"BURO_{e[0]}_{e[1].upper()}" for e in bureau_agg.columns.tolist()])
        
        # Domain ratios on aggregated bureau features
        if 'BURO_AMT_CREDIT_SUM_DEBT_SUM' in bureau_agg.columns and 'BURO_AMT_CREDIT_SUM_SUM' in bureau_agg.columns:
            bureau_agg['BURO_DEBT_CREDIT_RATIO'] = bureau_agg['BURO_AMT_CREDIT_SUM_DEBT_SUM'] / (bureau_agg['BURO_AMT_CREDIT_SUM_SUM'] + 1e-5)
            
        if 'BURO_AMT_CREDIT_SUM_OVERDUE_SUM' in bureau_agg.columns and 'BURO_AMT_CREDIT_SUM_SUM' in bureau_agg.columns:
            bureau_agg['BURO_OVERDUE_CREDIT_RATIO'] = bureau_agg['BURO_AMT_CREDIT_SUM_OVERDUE_SUM'] / (bureau_agg['BURO_AMT_CREDIT_SUM_SUM'] + 1e-5)
            
        bureau_agg = reduce_mem_usage(bureau_agg, verbose=False, logger=logger)
        del bureau
        gc.collect()
        
        logger.info(f"Bureau aggregated shape: {bureau_agg.shape}")
        return bureau_agg


def aggregate_previous_applications(project_root: Optional[str] = None, nan_as_category: bool = True) -> pd.DataFrame:
    """
    Process and aggregate previous_application.csv by SK_ID_CURR.
    """
    with timer("Previous Applications Aggregations", logger):
        prev_path = get_data_path("previous_application.csv", project_root)
        logger.info("Reading previous_application.csv...")
        prev = pd.read_csv(prev_path)
        prev = reduce_mem_usage(prev, verbose=False, logger=logger)
        
        # Replace anomalous days with NaN
        anom_cols = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
        for col in anom_cols:
            if col in prev.columns:
                prev[col] = prev[col].replace(365243, np.nan)
                
        # Ratios
        prev['APPLICATION_CREDIT_RATIO'] = prev['AMT_APPLICATION'] / (prev['AMT_CREDIT'] + 1e-5)
        prev['PAYMENT_RATE'] = prev['AMT_ANNUITY'] / (prev['AMT_CREDIT'] + 1e-5)
        prev['DOWN_PAYMENT_PERC'] = prev['AMT_DOWN_PAYMENT'] / (prev['AMT_CREDIT'] + 1e-5)
        
        # Categorical encoding
        cat_cols = [c for c in prev.columns if is_text_column(prev[c])]
        prev = pd.get_dummies(prev, columns=cat_cols, dummy_na=nan_as_category, dtype=float)
        
        # Aggregations
        num_aggregations = {
            'AMT_ANNUITY': ['min', 'max', 'mean', 'sum'],
            'AMT_APPLICATION': ['min', 'max', 'mean', 'sum'],
            'AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
            'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean', 'var'],
            'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
            'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
            'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'DAYS_DECISION': ['min', 'max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum', 'max'],
            'PAYMENT_RATE': ['min', 'max', 'mean'],
            'DOWN_PAYMENT_PERC': ['min', 'max', 'mean']
        }
        
        for c in prev.columns:
            if not c.startswith('SK_ID') and c not in num_aggregations and pd.api.types.is_numeric_dtype(prev[c]):
                num_aggregations[c] = ['mean', 'sum']
                
        valid_aggs = {k: v for k, v in num_aggregations.items() if k in prev.columns and pd.api.types.is_numeric_dtype(prev[k])}
        
        prev_agg = prev.groupby('SK_ID_CURR').agg(valid_aggs)
        prev_agg.columns = pd.Index([f"PREV_{e[0]}_{e[1].upper()}" for e in prev_agg.columns.tolist()])
        
        # Add count of previous applications
        prev_agg['PREV_APP_COUNT'] = prev.groupby('SK_ID_CURR').size()
        
        prev_agg = reduce_mem_usage(prev_agg, verbose=False, logger=logger)
        del prev
        gc.collect()
        
        logger.info(f"Previous applications aggregated shape: {prev_agg.shape}")
        return prev_agg


def aggregate_pos_cash(project_root: Optional[str] = None, nan_as_category: bool = True) -> pd.DataFrame:
    """
    Process and aggregate POS_CASH_balance.csv by SK_ID_CURR.
    """
    with timer("POS CASH Balance Aggregations", logger):
        pos_path = get_data_path("POS_CASH_balance.csv", project_root)
        logger.info("Reading POS_CASH_balance.csv...")
        pos = pd.read_csv(pos_path)
        pos = reduce_mem_usage(pos, verbose=False, logger=logger)
        
        cat_cols = [c for c in pos.columns if is_text_column(pos[c])]
        pos = pd.get_dummies(pos, columns=cat_cols, dummy_na=nan_as_category, dtype=float)
        
        num_aggregations = {
            'MONTHS_BALANCE': ['max', 'mean', 'size'],
            'SK_DPD': ['max', 'mean', 'sum'],
            'SK_DPD_DEF': ['max', 'mean', 'sum'],
            'CNT_INSTALMENT': ['max', 'mean'],
            'CNT_INSTALMENT_FUTURE': ['max', 'mean', 'min']
        }
        
        for c in pos.columns:
            if c.startswith('NAME_CONTRACT_STATUS_') and pd.api.types.is_numeric_dtype(pos[c]):
                num_aggregations[c] = ['mean', 'sum']
                
        valid_aggs = {k: v for k, v in num_aggregations.items() if k in pos.columns and pd.api.types.is_numeric_dtype(pos[k])}
        
        pos_agg = pos.groupby('SK_ID_CURR').agg(valid_aggs)
        pos_agg.columns = pd.Index([f"POS_{e[0]}_{e[1].upper()}" for e in pos_agg.columns.tolist()])
        
        pos_agg = reduce_mem_usage(pos_agg, verbose=False, logger=logger)
        del pos
        gc.collect()
        
        logger.info(f"POS CASH aggregated shape: {pos_agg.shape}")
        return pos_agg


def aggregate_installments_payments(project_root: Optional[str] = None) -> pd.DataFrame:
    """
    Process and aggregate installments_payments.csv by SK_ID_CURR.
    """
    with timer("Installments Payments Aggregations", logger):
        ins_path = get_data_path("installments_payments.csv", project_root)
        logger.info("Reading installments_payments.csv...")
        ins = pd.read_csv(ins_path)
        ins = reduce_mem_usage(ins, verbose=False, logger=logger)
        
        # Payment delay & differences
        ins['DPD'] = (ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']).clip(lower=0)
        ins['DBD'] = (ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']).clip(lower=0)
        ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / (ins['AMT_INSTALMENT'] + 1e-5)
        ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
        
        aggregations = {
            'DPD': ['max', 'mean', 'sum'],
            'DBD': ['max', 'mean', 'sum'],
            'PAYMENT_PERC': ['max', 'mean', 'min', 'var'],
            'PAYMENT_DIFF': ['max', 'mean', 'sum', 'min'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean'],
            'NUM_INSTALMENT_VERSION': ['nunique']
        }
        
        valid_aggs = {k: v for k, v in aggregations.items() if k in ins.columns and pd.api.types.is_numeric_dtype(ins[k])}
        
        ins_agg = ins.groupby('SK_ID_CURR').agg(valid_aggs)
        ins_agg.columns = pd.Index([f"INSTAL_{e[0]}_{e[1].upper()}" for e in ins_agg.columns.tolist()])
        ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
        
        ins_agg = reduce_mem_usage(ins_agg, verbose=False, logger=logger)
        del ins
        gc.collect()
        
        logger.info(f"Installments payments aggregated shape: {ins_agg.shape}")
        return ins_agg


def aggregate_credit_card_balance(project_root: Optional[str] = None) -> pd.DataFrame:
    """
    Process and aggregate credit_card_balance.csv by SK_ID_CURR.
    """
    with timer("Credit Card Balance Aggregations", logger):
        cc_path = get_data_path("credit_card_balance.csv", project_root)
        logger.info("Reading credit_card_balance.csv...")
        cc = pd.read_csv(cc_path)
        cc = reduce_mem_usage(cc, verbose=False, logger=logger)
        
        # Credit Card ratios
        cc['BALANCE_TO_LIMIT'] = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1e-5)
        cc['DRAWINGS_TO_LIMIT'] = cc['AMT_DRAWINGS_CURRENT'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1e-5)
        
        cat_cols = [c for c in cc.columns if is_text_column(cc[c])]
        cc = pd.get_dummies(cc, columns=cat_cols, dummy_na=True, dtype=float)
        
        aggregations = {
            'MONTHS_BALANCE': ['min', 'max', 'size'],
            'AMT_BALANCE': ['max', 'mean', 'sum'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean', 'sum'],
            'AMT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],
            'AMT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
            'AMT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'sum'],
            'AMT_INST_MIN_REGULARITY': ['max', 'mean', 'sum'],
            'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_RECEIVABLE_PRINCIPAL': ['max', 'mean'],
            'AMT_TOTAL_RECEIVABLE': ['max', 'mean'],
            'CNT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],
            'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
            'CNT_INSTALMENT_MATURE_CUM': ['max', 'mean'],
            'SK_DPD': ['max', 'mean', 'sum'],
            'SK_DPD_DEF': ['max', 'mean', 'sum'],
            'BALANCE_TO_LIMIT': ['max', 'mean'],
            'DRAWINGS_TO_LIMIT': ['max', 'mean']
        }
        
        for c in cc.columns:
            if c.startswith('NAME_CONTRACT_STATUS_') and pd.api.types.is_numeric_dtype(cc[c]):
                aggregations[c] = ['mean', 'sum']
                
        valid_aggs = {k: v for k, v in aggregations.items() if k in cc.columns and pd.api.types.is_numeric_dtype(cc[k])}
        
        cc_agg = cc.groupby('SK_ID_CURR').agg(valid_aggs)
        cc_agg.columns = pd.Index([f"CC_{e[0]}_{e[1].upper()}" for e in cc_agg.columns.tolist()])
        
        cc_agg = reduce_mem_usage(cc_agg, verbose=False, logger=logger)
        del cc
        gc.collect()
        
        logger.info(f"Credit card balance aggregated shape: {cc_agg.shape}")
        return cc_agg


def build_full_feature_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    project_root: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs all domain features and merges all secondary table aggregations.
    """
    with timer("Full Feature Matrix Construction", logger):
        # 1. Application Domain Features
        logger.info("Building application domain features...")
        train_df = build_application_domain_features(train_df)
        test_df = build_application_domain_features(test_df)
        
        # 2. Bureau & Bureau Balance
        bureau_agg = aggregate_bureau_and_balance(project_root)
        train_df = train_df.merge(bureau_agg, how='left', on='SK_ID_CURR')
        test_df = test_df.merge(bureau_agg, how='left', on='SK_ID_CURR')
        del bureau_agg
        gc.collect()
        
        # 3. Previous Applications
        prev_agg = aggregate_previous_applications(project_root)
        train_df = train_df.merge(prev_agg, how='left', on='SK_ID_CURR')
        test_df = test_df.merge(prev_agg, how='left', on='SK_ID_CURR')
        del prev_agg
        gc.collect()
        
        # 4. POS CASH Balance
        pos_agg = aggregate_pos_cash(project_root)
        train_df = train_df.merge(pos_agg, how='left', on='SK_ID_CURR')
        test_df = test_df.merge(pos_agg, how='left', on='SK_ID_CURR')
        del pos_agg
        gc.collect()
        
        # 5. Installments Payments
        ins_agg = aggregate_installments_payments(project_root)
        train_df = train_df.merge(ins_agg, how='left', on='SK_ID_CURR')
        test_df = test_df.merge(ins_agg, how='left', on='SK_ID_CURR')
        del ins_agg
        gc.collect()
        
        # 6. Credit Card Balance
        cc_agg = aggregate_credit_card_balance(project_root)
        train_df = train_df.merge(cc_agg, how='left', on='SK_ID_CURR')
        test_df = test_df.merge(cc_agg, how='left', on='SK_ID_CURR')
        del cc_agg
        gc.collect()
        
        # Clean column names for LightGBM
        logger.info("Sanitizing feature column names and values...")
        train_df = train_df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))
        test_df = test_df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))
        
        train_df = train_df.replace([np.inf, -np.inf], np.nan)
        test_df = test_df.replace([np.inf, -np.inf], np.nan)
        
        train_df = reduce_mem_usage(train_df, verbose=True, logger=logger)
        test_df = reduce_mem_usage(test_df, verbose=True, logger=logger)
        
        logger.info(f"Final Train shape: {train_df.shape}, Final Test shape: {test_df.shape}")
        
    return train_df, test_df

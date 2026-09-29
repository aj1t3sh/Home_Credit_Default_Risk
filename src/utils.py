"""
Utility functions for memory optimization, logging, and environment configuration.
"""

import gc
import logging
import os
import sys
import time
from contextlib import contextmanager
import numpy as np
import pandas as pd


def get_logger(name: str = "home_credit") -> logging.Logger:
    """
    Configure and return a structured logger with both console and file handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler in reports directory
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "pipeline.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


@contextmanager
def timer(title: str, logger: logging.Logger = None):
    """
    Context manager to profile execution time of code blocks.
    """
    t0 = time.time()
    if logger:
        logger.info(f"==> Starting [{title}]...")
    else:
        print(f"==> Starting [{title}]...")
    yield
    elapsed = time.time() - t0
    if logger:
        logger.info(f"==> Finished [{title}] in {elapsed:.2f}s ({elapsed/60:.2f} min)")
    else:
        print(f"==> Finished [{title}] in {elapsed:.2f}s ({elapsed/60:.2f} min)")


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True, logger: logging.Logger = None) -> pd.DataFrame:
    """
    Iterate through all columns of a dataframe and modify the data type
    to reduce memory usage safely without losing precision.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        c_min = df[col].min()
        c_max = df[col].max()
        
        if pd.isna(c_min) or pd.isna(c_max):
            # Column has all NaNs or is float with NaNs
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype(np.float32)
            continue
            
        if pd.api.types.is_integer_dtype(df[col]):
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        elif pd.api.types.is_float_dtype(df[col]):
            # Use float32 for stable tree boosting & lower memory
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
                
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    msg = f"Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB (-{reduction:.1f}% reduction)"
    if verbose:
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
    return df


def get_data_path(filename: str, project_root: str = None) -> str:
    """
    Resolve data file path checking both data/raw/ and root project directory.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    path1 = os.path.join(project_root, "data", "raw", filename)
    if os.path.exists(path1):
        return path1
        
    path2 = os.path.join(project_root, filename)
    if os.path.exists(path2):
        return path2
        
    raise FileNotFoundError(f"Could not locate dataset file: {filename} in data/raw/ or project root.")

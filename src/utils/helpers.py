import pandas as pd
import numpy as np
from typing import Tuple
from config.settings import model_config

def create_time_based_split(df: pd.DataFrame, date_col: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by year (time-based)"""
    
    df['year'] = pd.to_datetime(df[date_col]).dt.year
    
    train = df[df['year'] <= model_config.TRAIN_END_YEAR].copy()
    val = df[df['year'] == model_config.VAL_YEAR].copy()
    test = df[df['year'] >= model_config.TEST_START_YEAR].copy()
    
    return train, val, test

def calculate_class_weights(y):
    """Calculate balanced class weights"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )
    
    return dict(zip(classes, weights))
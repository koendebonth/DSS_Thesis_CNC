#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved experiment script without data leakage.
Allows specifying machine adoption percentages (M01, M02, M03).
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for module imports
project_root = Path(__file__).resolve().parent.parent
# Insert project root and utils folder on sys.path to ensure package discovery
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'utils'))

import os
import logging
from collections import Counter

# Standard libraries
import numpy as np
import pandas as pd

# Machine learning libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Imbalanced-learn for SMOTE
from imblearn.over_sampling import SMOTE

# Custom utilities
from utils.data_loader_utils import load_tool_research_data
from utils.feature_extraction import transform_data

# Configure logging to output to both file and console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatters and handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handler
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)
file_handler = logging.FileHandler(log_dir / f'experiment_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(exclude_processes=None):
    """
    Load raw vibration data for all machines and processes.
    Returns raw X_data (list of ndarrays) and y_data (list of labels).
    """
    machines = ["M01", "M02", "M03"]
    processes = [f"OP{i:02d}" for i in range(15)]
    labels = ["good", "bad"]

    if exclude_processes:
        processes = [p for p in processes if p not in exclude_processes]

    data_dir = project_root / "data"
    X_data, y_data = [], []
    for machine in machines:
        for proc in processes:
            for lab in labels:
                path = data_dir / machine / proc / lab
                samples, sample_labels = load_tool_research_data(path, label=lab)
                X_data.extend(samples)
                y_data.extend(sample_labels)
    # trace: log raw data loading stats
    logger.info(f"Loaded raw data: {len(X_data)} samples; machines: {pd.Series([lbl.split('_')[0] for lbl in y_data]).value_counts().to_dict()}; labels: {pd.Series([lbl.split('_')[-1] for lbl in y_data]).value_counts().to_dict()}")
    return X_data, y_data


def split_by_adoption(X_data, y_data, M01, M02, M03, random_state=42):
    """
    Split raw data into train and test sets based on machine adoption percentages.
    M01, M02, M03 are floats in [0,1] indicating fraction of that machine's data for training.
    """
    df = pd.DataFrame({'data': X_data, 'label': y_data})
    df[['machine','month','year','process','sample_id','status']] = df['label'].str.split("_", expand=True)
    # trace: log split parameters
    logger.info(f"Splitting data with adoption fractions: M01={M01}, M02={M02}, M03={M03}")
    train_frames, test_frames = [], []
    for machine, pct in zip(["M01", "M02", "M03"], [M01, M02, M03]):
        subset = df[df['machine'] == machine]
        if pct > 0:
            train_sub = subset.sample(frac=pct, random_state=random_state)
        else:
            train_sub = subset.iloc[0:0]
        test_sub = subset.drop(train_sub.index)
        train_frames.append(train_sub)
        test_frames.append(test_sub)

    train_df = pd.concat(train_frames).reset_index(drop=True)
    test_df = pd.concat(test_frames).reset_index(drop=True)
    # trace: log train/test sizes
    logger.info(f"Train samples: {len(train_df)}; per machine: {train_df['machine'].value_counts().to_dict()}; Test samples: {len(test_df)}; per machine: {test_df['machine'].value_counts().to_dict()}")
    # trace: log class distribution in train set
    logger.info(f"Train class distribution: {train_df['status'].value_counts().to_dict()}")
    logger.info(f"Train class distribution per machine: {train_df.groupby('machine')['status'].value_counts().unstack(fill_value=0).to_dict('index')}")
    
    # Extract raw lists
    X_train_raw = train_df['data'].tolist()
    y_train_raw = train_df['label'].tolist()
    X_test_raw  = test_df['data'].tolist()
    y_test_raw  = test_df['label'].tolist()

    return X_train_raw, X_test_raw, y_train_raw, y_test_raw


def run_holdout_experiment(X_data=None, y_data=None, M01=1.0, M02=0.0, M03=0.0, machines_to_reduce=None, random_state=42, check_split=True):
    """
    Run a holdout experiment with specified machine adoption parameters.
    """
    # Load raw data if not provided
    if X_data is None or y_data is None:
        X_data, y_data = load_data()

    # Split before any feature extraction
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_by_adoption(
        X_data, y_data, M01, M02, M03, random_state=random_state)
    # Undersample training set if requested
    if machines_to_reduce:
        logger.info(f"Undersampling good class for machines: {machines_to_reduce}")
        X_train_raw, y_train_raw = undersample_good_class_to_minimum(X_train_raw, y_train_raw, machines_to_reduce, random_state=random_state)
        logger.info(f"After undersampling, train samples: {len(y_train_raw)}; per machine: {pd.Series([lbl.split('_')[0] for lbl in y_train_raw]).value_counts().to_dict()}")

    # Feature extraction on train and test separately
    # trace: start feature extraction
    logger.info("Starting feature extraction on train set")
    X_train, y_train = transform_data(X_train_raw, y_train_raw, include_metadata=False)
    X_test,  y_test  = transform_data(X_test_raw,  y_test_raw,  include_metadata=False)
    # trace: log feature shapes
    logger.info(f"Feature shapes - train: {X_train.shape}, test: {X_test.shape}")

    # SMOTE oversampling on training data only
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    # trace: log SMOTE results
    logger.info(f"SMOTE applied: resampled train size: {len(y_train_res)}; class distribution: {pd.Series(y_train_res).value_counts().to_dict()}")
    logger.info(f"Feature shapes after SMOTE - train: {X_train_res.shape}")

    # Train the Random Forest classifier
    clf = RandomForestClassifier(
        max_features='log2',
        n_estimators=150,
        max_depth=15,
        random_state=random_state
    )
    clf.fit(X_train_res, y_train_res)
    # trace: training complete
    logger.info("Random Forest training complete")

    # Evaluate on test set
    y_pred = clf.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['bad','good']))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    return clf


# Add function to undersample good class in training set only
def undersample_good_class_to_minimum(X_data, y_data, machines_to_reduce, random_state=42):
    """
    Undersample the 'good' class for specified machines within the training set so that each process has equal minimum count.
    """
    import pandas as pd
    df = pd.DataFrame({'data': X_data, 'label': y_data})
    df[['machine','month','year','process','sample_id','status']] = df['label'].str.split("_", expand=True)
    good = df[df['status']=='good']
    bad = df[df['status']=='bad']
    if isinstance(machines_to_reduce, str):
        machines_to_reduce = [machines_to_reduce]
    reduced_good = pd.DataFrame()
    for m in machines_to_reduce:
        sub = good[good['machine']==m]
        if sub.empty:
            continue
        counts = sub['process'].value_counts()
        mcount = counts.min()
        for proc in counts.index:
            sampled = sub[sub['process']==proc].sample(n=mcount, random_state=random_state)
            reduced_good = pd.concat([reduced_good, sampled], axis=0)
    other_good = good[~good['machine'].isin(machines_to_reduce)]
    final_df = pd.concat([reduced_good, other_good, bad], axis=0).reset_index(drop=True)
    return final_df['data'].tolist(), final_df['label'].tolist()


# Add function to run experiment using preloaded data and avoid leakage
def run_holdout_experiment_preloaded(X_data, y_data, M01=1.0, M02=0.0, M03=0.0, machines_to_reduce=None, random_state=42, check_split=True):
    """
    Run a holdout experiment on already loaded data, with optional undersampling on the training set to prevent data leakage.
    """
    # Split raw data
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_by_adoption(X_data, y_data, M01, M02, M03, random_state=random_state)
    # Optionally undersample training set
    if machines_to_reduce:
        logger.info(f"Undersampling good class for machines: {machines_to_reduce}")
        X_train_raw, y_train_raw = undersample_good_class_to_minimum(X_train_raw, y_train_raw, machines_to_reduce, random_state=random_state)
    # Feature extraction
    logger.info("Starting feature extraction on train set")
    X_train, y_train = transform_data(X_train_raw, y_train_raw, include_metadata=False)
    X_test, y_test = transform_data(X_test_raw, y_test_raw, include_metadata=False)
    logger.info(f"Feature shapes - train: {X_train.shape}, test: {X_test.shape}")
    # SMOTE on train data only
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(f"SMOTE applied: resampled train size: {len(y_train_res)}; class distribution: {pd.Series(y_train_res).value_counts().to_dict()}")
    # Train model
    clf = RandomForestClassifier(max_features='log2', n_estimators=150, max_depth=15, random_state=random_state)
    clf.fit(X_train_res, y_train_res)
    logger.info("Random Forest training complete")
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['bad','good']))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    return clf


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--M01', type=float, default=1.0, help='Fraction of M01 data for training')
    parser.add_argument('--M02', type=float, default=0.0, help='Fraction of M02 data for training')
    parser.add_argument('--M03', type=float, default=0.0, help='Fraction of M03 data for training')
    parser.add_argument('--check-split', action='store_true', help='Enable verification of split_by_adoption correctness')
    args = parser.parse_args()

    run_holdout_experiment(args.M01, args.M02, args.M03, check_split=args.check_split) 
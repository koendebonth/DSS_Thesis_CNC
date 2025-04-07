from sklearn.model_selection import BaseCrossValidator
import numpy as np
from collections import defaultdict
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

class MachineSplitCV(BaseCrossValidator):
    """Custom cross-validator that keeps samples from same machine/operation together"""
    
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        
    def split(self, X, y=None, groups=None):
        # Extract machine and operation info from labels
        labels = np.array([label.split('_') for label in y])
        machines = labels[:, 0]  # First element is machine
        operations = labels[:, 1]  # Second element is operation
        
        # Create unique machine-operation combinations
        machine_op_combinations = np.unique(list(zip(machines, operations)), axis=0)
        np.random.shuffle(machine_op_combinations)
        
        # Split combinations into folds
        fold_size = len(machine_op_combinations) // self.n_splits
        folds = [machine_op_combinations[i:i + fold_size] for i in range(0, len(machine_op_combinations), fold_size)]
        
        # Generate train/test indices for each fold
        for i in range(self.n_splits):
            test_combinations = folds[i]
            test_mask = np.zeros(len(y), dtype=bool)
            
            # Mark samples for test set
            for machine, operation in test_combinations:
                test_mask |= (machines == machine) & (operations == operation)
            
            train_mask = ~test_mask
            yield np.where(train_mask)[0], np.where(test_mask)[0]
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

def perform_cross_validation(model, X_data, y_data, n_splits=5, random_state=42):
    """
    Perform cross-validation with proper handling of SMOTE and feature extraction
    
    Parameters:
    -----------
    model : estimator object
        The model to evaluate
    X_data : list
        Raw input data
    y_data : list
        Labels
    n_splits : int
        Number of cross-validation splits
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    dict : Dictionary containing evaluation metrics for each fold
    """
    cv = MachineSplitCV(n_splits=n_splits)
    fold_results = defaultdict(list)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_data, y_data)):
        # Split data
        X_train = [X_data[i] for i in train_idx]
        X_test = [X_data[i] for i in test_idx]
        y_train = [y_data[i] for i in train_idx]
        y_test = [y_data[i] for i in test_idx]
        
        # Extract features (assuming transform_data is imported)
        from utils.feature_extraction import transform_data
        X_train_features, y_train_labels = transform_data(X_train, y_train, include_metadata=False)
        X_test_features, y_test_labels = transform_data(X_test, y_test, include_metadata=False)
        
        # Apply SMOTE only on training data
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_features, y_train_labels)
        
        # Train model
        model_clone = clone(model)
        model_clone.fit(X_train_resampled, y_train_resampled)
        
        # Predict
        y_pred = model_clone.predict(X_test_features)
        
        # Store results
        fold_results['fold'].append(fold_idx)
        fold_results['train_samples'].append(len(y_train_resampled))
        fold_results['test_samples'].append(len(y_test_labels))
        
        # Calculate metrics
        report = classification_report(y_test_labels, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test_labels, y_pred)
        
        # Store metrics
        for label in ['good', 'bad']:
            if label in report:
                metrics = report[label]
                fold_results[f'{label}_precision'].append(metrics['precision'])
                fold_results[f'{label}_recall'].append(metrics['recall'])
                fold_results[f'{label}_f1'].append(metrics['f1-score'])
        
        fold_results['accuracy'].append(report['accuracy'])
        fold_results['confusion_matrix'].append(conf_matrix)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(fold_results)
    return results_df 
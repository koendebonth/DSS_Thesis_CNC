import numpy as np
import pandas as pd
import pywt
from scipy import stats
from tqdm import tqdm

def extract_wavelet_features(signal, wavelet='coif8', max_level=3):
    """
    Perform wavelet packet decomposition and extract statistical features
    
    Parameters:
    -----------
    signal : ndarray
        Input signal (1D array)
    wavelet : str
        Wavelet to use (default: 'coif8')
    max_level : int
        Maximum decomposition level
        
    Returns:
    --------
    dict: Dictionary of statistical features
    """
    # Create wavelet packet
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
    
    # Extract nodes at the maximum level
    level_nodes = [node.path for node in wp.get_level(max_level, 'natural')]
    
    features = {}
    
    for node in level_nodes:
        # Get coefficients for this node
        coeffs = wp[node].data
        
        # Extract statistical features
        features[f"mean_{node}"] = np.mean(coeffs)
        features[f"max_{node}"] = np.max(coeffs)
        features[f"min_{node}"] = np.min(coeffs)
        features[f"std_{node}"] = np.std(coeffs)
        features[f"kurtosis_{node}"] = stats.kurtosis(coeffs)
        features[f"skewness_{node}"] = stats.skew(coeffs)
        
        # Shannon entropy
        # Normalize the coefficients
        coeffs_norm = np.abs(coeffs) / np.sum(np.abs(coeffs) + 1e-10)
        entropy = -np.sum(coeffs_norm * np.log2(coeffs_norm + 1e-10))
        features[f"entropy_{node}"] = entropy
        
    return features

def transform_data(X_data, y_data, include_metadata=False, wavelet='coif8', max_level=3):
    """
    Transform time series data using wavelet packet transform and extract features
    
    Parameters:
    -----------
    X_data : list of ndarrays
        List containing signal data
    y_data : list
        List containing labels for each signal
    include_metadata : bool, default=False
        Whether to include metadata columns (machine, process) in output
    wavelet : str, default='coif8'
        Wavelet to use for transform
    max_level : int, default=3
        Maximum decomposition level
        
    Returns:
    --------
    X : DataFrame
        Features dataframe
    y : Series
        Labels series
    """
    # List to store all features
    all_features = []

    # Process each sample in X_data
    for i, sample in enumerate(tqdm(X_data, desc="Extracting features")):
        sample_features = {}
        
        # Process each axis (channel)
        for axis in range(sample.shape[1]):
            # Get signal for this axis
            signal = sample[:, axis]
            
            # Apply wavelet packet transform and extract features
            wp_features = extract_wavelet_features(signal, wavelet=wavelet, max_level=max_level)
            
            # Add axis identifier to feature names
            for key, value in wp_features.items():
                sample_features[f"axis{axis}_{key}"] = value
        
        # Add to collection
        all_features.append(sample_features)

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Extract label information
    labels = []
    machines = []
    processes = []
    
    for label in y_data:
        split_label = label.split("_")
        labels.append(split_label[-1])
        machines.append(split_label[0])
        processes.append(split_label[1])
    
    # Create output variables
    y = pd.Series(labels)
    
    # If metadata should be included
    if include_metadata:
        features_df['machine'] = machines
        features_df['process'] = processes
        
    X = features_df
    
    print("\u2705 Feature extraction completed successfully!")
    
    return X, y

def prepare_train_test_data(X_data, y_data, test_size=0.2, random_state=42, prevent_leakage=True):
    """
    Full data preparation pipeline: extract features and split into train/test
    
    Parameters:
    -----------
    X_data : list of ndarrays
        List containing signal data
    y_data : list
        List containing labels for each signal
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    prevent_leakage : bool, default=True
        Apply measures to prevent data leakage
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : DataFrames and Series
        Train and test datasets
    """
    from sklearn.model_selection import train_test_split
    
    # First extract features with metadata for stratification
    X_features, y_labels = transform_data(X_data, y_data, include_metadata=True)
    
    # Create stratification variable if needed
    if prevent_leakage:
        # Stratify by both machine and process to prevent leakage
        stratify_by = X_features['machine'] + "_" + X_features['process'] + "_" + y_labels
    else:
        stratify_by = y_labels
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_by
    )
    
    # Remove metadata columns if leakage prevention is enabled
    if prevent_leakage:
        metadata_cols = ['machine', 'process']
        X_train = X_train.drop(columns=metadata_cols)
        X_test = X_test.drop(columns=metadata_cols)
    
    return X_train, X_test, y_train, y_test
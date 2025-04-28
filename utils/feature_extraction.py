import numpy as np
import pandas as pd
import pywt
from scipy import stats
from tqdm import tqdm
from typing import Tuple
from tqdm.contrib.concurrent import process_map

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
    # Determine and clamp the actual decomposition level
    max_lvl_allowed = pywt.dwt_max_level(len(signal), wavelet)
    level = min(max_level, max_lvl_allowed)
    # Create wavelet packet
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=level)
    # Extract nodes at the actual level
    level_nodes = [node.path for node in wp.get_level(level, 'natural')]
    
    features = {}
    
    for node in level_nodes:
        # Get coefficients for this node
        coeffs = wp[node].data
        
        # Extract statistical features
        features[f"mean_{node}"] = np.mean(coeffs)
        features[f"max_{node}"] = np.max(coeffs)
        features[f"min_{node}"] = np.min(coeffs)
        features[f"std_{node}"] = np.std(coeffs)
        k = stats.kurtosis(coeffs)
        features[f"kurtosis_{node}"] = 0 if np.isnan(k) else k
        s = stats.skew(coeffs)
        features[f"skewness_{node}"] = 0 if np.isnan(s) else s
        
        # Shannon entropy
        # Normalize the coefficients
        denom = np.sum(np.abs(coeffs)) + 1e-10
        coeffs_norm = np.abs(coeffs) / denom
        entropy = -np.sum(coeffs_norm * np.log2(coeffs_norm + 1e-10))
        features[f"entropy_{node}"] = entropy
        
    return features

def transform_data(X_data, y_data,label_type='binary', include_metadata=False, wavelet='coif8', max_level=3) -> Tuple[pd.DataFrame, pd.Series]:
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
        Features dataframe (plus 'machine' and 'process' columns if include_metadata=True)
    y : Series
        Labels series (1 for 'good', 0 for 'bad')
    """
    # Basic input validation
    assert len(X_data) == len(y_data), "X_data and y_data must have the same length"
    for i, sample in enumerate(X_data):
        assert hasattr(sample, 'ndim') and sample.ndim == 2, f"Sample {i} is not a 2D array"

    # Extract features in parallel with progress bar
    all_features = process_map(
        lambda sample: {
            f"axis{axis}_{key}": value
            for axis in range(sample.shape[1])
            for key, value in extract_wavelet_features(sample[:, axis], wavelet=wavelet, max_level=max_level).items()
        },
        X_data,
        desc="Extracting features"
    )

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    if label_type != "binary":
        # Extract label information
        labels = []
        machines = []
        processes = []
        
        for label in y_data:
            split = label.split("_")
            labels.append(1 if split[-1] == 'good' else 0)
            machines.append(split[0])
            processes.append(split[3])
        
        # Create output variables
        y = pd.Series(labels)
    else:
        labels = []
        for label in y_data:
            labels.append(label)

        y = pd.Series(labels) 

    
    X = features_df
    
    # If metadata should be included
    if include_metadata:
        features_df['machine'] = machines
        features_df['process'] = processes
        
    X = features_df
    
    print("\u2705 Feature extraction completed successfully!")
    
    return X, y
import os
import sys
from pathlib import Path
import data_loader_utils
import itertools 
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import numpy as np
from scipy import signal,stats
from tqdm import tqdm

# Get the current working directory
current_dir = os.getcwd()

machines = ["M01","M02","M03"]
process_names = ["OP00","OP01","OP02","OP03","OP04","OP05","OP06","OP07","OP08","OP09","OP10","OP11","OP12","OP13","OP14"]
labels = ["good","bad"]

path_to_dataset = os.path.join(current_dir, "data")

# List to store metadata about each h5 file
file_metadata = []

# Sampling rate in Hz for the measurements
SAMPLING_RATE = 2000  

for process_name, machine, label in itertools.product(process_names, machines, labels):
    data_path = os.path.join(path_to_dataset, machine, process_name, label)
    if os.path.exists(data_path):
        for file in os.listdir(data_path):
            if file.endswith('.h5'):
                fullpath = os.path.join(data_path, file)
                with h5py.File(fullpath, 'r') as f:
                    dataset_name = list(f.keys())[0]
                    dataset = f[dataset_name]
                    
                    # Calculate durations and round to whole seconds/minutes
                    duration_sec = round(dataset.shape[0] / SAMPLING_RATE)  # Round to whole seconds
                    duration_min = round(dataset.shape[0] / SAMPLING_RATE / 60, 2)  # Keep 2 decimal places for minutes
                    
                    file_metadata.append({
                        'machine': str(machine),
                        'operation': str(process_name),
                        'class': str(label),
                        'measurements': int(dataset.shape[0]),
                        'channels': int(dataset.shape[1] if len(dataset.shape) > 1 else 1),
                        'duration_sec': duration_sec,  # Rounded to whole seconds
                        'duration_min': duration_min,  # Rounded to 2 decimal places
                        'file_size_mb': round(os.path.getsize(fullpath) / (1024 * 1024), 2),  # Also round file size
                        'month_created': str(file.split('_')[1]),
                        'year_created': str(file.split('_')[2]),
                        'full_path': str(fullpath)
                    })

# Create dataframe with file metadata
df_measurement_files = pd.DataFrame(file_metadata)

# Ensure correct data types
df_measurement_files = df_measurement_files.astype({
    'machine': 'str',
    'operation': 'str',
    'class': 'str',
    'measurements': 'int',
    'channels': 'int',
    'duration_sec': 'float',
    'duration_min': 'float',
    'file_size_mb': 'float',
    'month_created': 'str',
    'year_created': 'str',
    'full_path': 'str'
})

df_measurement_files.to_csv('export/measurement_files_metadata.csv', index=False)

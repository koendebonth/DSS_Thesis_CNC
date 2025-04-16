import os
import h5py
import pandas as pd
import numpy as np
import itertools

# Constants
SAMPLING_RATE = 2000  # Sampling rate in Hz for the measurements
MACHINES = ["M01", "M02", "M03"]
PROCESS_NAMES = ["OP00", "OP01", "OP02", "OP03", "OP04", "OP05", "OP06", 
                "OP07", "OP08", "OP09", "OP10", "OP11", "OP12", "OP13", "OP14"]
LABELS = ["good", "bad"]

def collect_file_metadata():
    """
    Collect metadata from all measurement files in the dataset.
    
    Returns:
        pd.DataFrame: DataFrame containing metadata for all files
    """
    # Get the current working directory
    current_dir = os.getcwd()
    path_to_dataset = os.path.join(current_dir, "data")
    
    # List to store metadata about each h5 file
    file_metadata = []

    for process_name, machine, label in itertools.product(PROCESS_NAMES, MACHINES, LABELS):
        data_path = os.path.join(path_to_dataset, machine, process_name, label)
        if os.path.exists(data_path):
            for file in os.listdir(data_path):
                if file.endswith('.h5'):
                    fullpath = os.path.join(data_path, file)
                    with h5py.File(fullpath, 'r') as f:
                        dataset_name = list(f.keys())[0]
                        dataset = f[dataset_name]
                        
                        # Calculate durations and round to whole seconds/minutes
                        duration_sec = round(dataset.shape[0] / SAMPLING_RATE)
                        duration_min = round(dataset.shape[0] / SAMPLING_RATE / 60, 2)
                        
                        file_metadata.append({
                            'machine': str(machine),
                            'operation': str(process_name),
                            'class': str(label),
                            'measurements': int(dataset.shape[0]),
                            'channels': int(dataset.shape[1] if len(dataset.shape) > 1 else 1),
                            'duration_sec': duration_sec,
                            'duration_min': duration_min,
                            'file_size_mb': round(os.path.getsize(fullpath) / (1024 * 1024), 2),
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
    
    return df_measurement_files

def main():
    """Main function to collect and export metadata."""
    # Collect metadata
    df_measurement_files = collect_file_metadata()
    
    # Create export directory if it doesn't exist
    current_dir = os.getcwd()
    export_dir = os.path.join(current_dir, 'export')
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        print(f"Export directory created at: {export_dir}")
    
    # Export metadata to CSV
    df_measurement_files.to_csv('export/measurement_files_metadata.csv', index=False)
    print("Metadata successfully exported to CSV file.")

if __name__ == "__main__":
    main()

import os
import itertools
from tqdm import tqdm
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow importing from utils
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils import data_loader_utils

def load_data(exclude_processes=None):
    """
    Load data from all machines and processes, with option to exclude specific processes.

    Args:
        exclude_processes (list, optional): List of process names to exclude from loading.

    Returns:
        tuple: (X_data, y_data, y_binary) containing features, full labels, and binary labels
    """
    machines = ["M01","M02","M03"]
    process_names = ["OP00","OP01","OP02","OP03","OP04","OP05","OP06","OP07","OP08","OP09","OP10","OP11","OP12","OP13","OP14"]
    labels = ["good","bad"]
    
    # Filter out excluded processes if any
    if exclude_processes:
        process_names = [p for p in process_names if p not in exclude_processes]
    
    path_to_dataset = os.path.join(root_dir, "data")
    
    X_data = []
    y_data = []
    
    try:
        # Calculate total number of combinations
        total_combinations = len(process_names) * len(machines) * len(labels)
        
        # Create progress bar
        with tqdm(total=total_combinations, desc="Loading data") as pbar:
            for process_name, machine, label in itertools.product(process_names, machines, labels):
                data_path = os.path.join(path_to_dataset, machine, process_name, label)
                data_list, data_label = data_loader_utils.load_tool_research_data(data_path, label=label)
                X_data.extend(data_list)
                y_data.extend(data_label)
                pbar.update(1)
                pbar.set_postfix({"Samples": len(X_data)})
                
        print(f"Data loaded successfully ✅ - {len(X_data)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
    
    # Generate binary labels from full label strings
    y_binary = [0 if label_str.split("_")[-1] == "good" else 1 for label_str in y_data]

    return X_data, y_data, y_binary
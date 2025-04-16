import pandas as pd
from typing import Dict, Union
import numpy as np

def classification_report_to_latex(
    report: Union[str, Dict],
    caption: str = "Classification Report",
    label: str = "tab:classification_report",
    precision: int = 3
) -> str:
    """
    Convert a scikit-learn classification report to a LaTeX table.
    
    Args:
        report: Either a string classification report from scikit-learn or a dictionary
               containing the classification metrics
        caption: Table caption
        label: Table label for LaTeX referencing
        precision: Number of decimal places to round the metrics
        
    Returns:
        str: LaTeX table code
    """
    # Convert string report to dictionary if needed
    if isinstance(report, str):
        report_dict = {}
        lines = report.split('\n')
        for line in lines[2:-1]:  # Skip header and footer
            if line.strip():
                row = line.split()
                if len(row) > 1:
                    class_name = row[0]
                    metrics = [float(x) for x in row[1:]]
                    report_dict[class_name] = metrics
    else:
        report_dict = report

    # Convert to DataFrame
    df = pd.DataFrame(report_dict).T
    df.columns = ['Precision', 'Recall', 'F1-score', 'Support']
    
    # Round values
    df = df.round(precision)
    
    # Generate LaTeX table
    latex_table = f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{l{'r' * len(df.columns)}}}
\\toprule
Class & {' & '.join(df.columns)} \\\\
\\midrule
"""
    
    # Add data rows
    for idx, row in df.iterrows():
        latex_table += f"{idx} & {' & '.join(row.astype(str))} \\\\\n"
    
    # Add bottom rule and end table
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex_table 
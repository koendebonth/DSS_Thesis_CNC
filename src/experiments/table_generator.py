import pandas as pd

def create_n_star_table(df_all, f1_threshold=0.914):
    """
    Creates a table showing the minimum k where F1 > f1_threshold for each source-target pair.
    
    Parameters:
    -----------
    df_all : pd.DataFrame
        DataFrame with columns: 'source', 'target', 'k', 'rep', 'f1_macro'
    f1_threshold : float, default=0.914
        F1 score threshold to find the minimum k
        
    Returns:
    --------
    pd.DataFrame
        Pivot table with sources as rows, targets as columns, and n★ values in cells
    """
    n_star_results = []

    # Group by source-target pairs and k, calculate median F1 score
    for (src, tgt), group in df_all.groupby(["source", "target"]):
        k_medians = group.groupby("k")["f1_macro"].median().reset_index()
        
        # Find the first k where F1 > threshold
        qualifying_k = k_medians[k_medians["f1_macro"] > f1_threshold]
        if not qualifying_k.empty:
            min_k = qualifying_k["k"].min()
            f1_at_min_k = qualifying_k.loc[qualifying_k["k"] == min_k, "f1_macro"].values[0]
            n_star_results.append({"source": src, "target": tgt, "n_star": min_k, "f1_score": f1_at_min_k})
        else:
            n_star_results.append({"source": src, "target": tgt, "n_star": None, "f1_score": None})

    # Create DataFrame and format as a pivot table
    n_star_df = pd.DataFrame(n_star_results)
    n_star_pivot = n_star_df.pivot(index="source", columns="target", values="n_star")
    
    return n_star_pivot

# Example usage:
# n_star_table = create_n_star_table(df_all)
# display(n_star_table.style.set_caption("n★ values where F1 score exceeds 0.914")) 
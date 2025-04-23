#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA script: compute correlations, scatter plots, and 2D embeddings (t-SNE & UMAP) of features.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Attempt to import UMAP
try:
    import umap
    has_umap = True
except ImportError:
    has_umap = False

# Ensure project root for module imports via EXP_improved
sys.path.insert(0, str(Path(__file__).resolve().parent))
from EXP_improved import load_data, transform_data


def main():
    # Load and transform data
    X_data, y_data = load_data()
    X, y = transform_data(X_data, y_data, include_metadata=False)

    # Prepare DataFrame
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['status'] = y
    color_map = {'good': 'green', 'bad': 'red'}

    # Create results directory
    results_dir = Path(__file__).resolve().parent / 'results'
    results_dir.mkdir(exist_ok=True)

    # Correlation heatmap
    corr = df[feature_names].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(results_dir / 'correlation_heatmap.png')
    plt.close()

    # Scatter matrix
    sns.set(style='ticks')
    pd.plotting.scatter_matrix(
        df[feature_names], alpha=0.5, figsize=(12, 12), diagonal='kde',
        c=df['status'].map(color_map)
    )
    plt.suptitle('Scatter Matrix of Features Colored by Status', y=1.02)
    plt.savefig(results_dir / 'scatter_matrix.png')
    plt.close()

    # t-SNE embedding
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['status'].map(color_map), alpha=0.7)
    plt.title('t-SNE of Features')
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.tight_layout()
    plt.savefig(results_dir / 'tsne_plot.png')
    plt.close()

    # UMAP embedding (if available)
    if has_umap:
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X)
        plt.figure(figsize=(8, 6))
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=df['status'].map(color_map), alpha=0.7)
        plt.title('UMAP of Features')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.tight_layout()
        plt.savefig(results_dir / 'umap_plot.png')
        plt.close()
    else:
        print("UMAP not installed. Please install umap-learn to generate UMAP plots.")

    print(f"EDA plots saved in {results_dir}")


if __name__ == '__main__':
    main() 
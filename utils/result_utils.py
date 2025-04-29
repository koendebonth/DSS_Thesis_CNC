import pandas as pd
import numpy as np

def create_results_df():
    """
    Initialize an empty DataFrame to store experiment results:
      – M01_pct, M02_pct, M03_pct: fractions used in the train split
      – train_normals, train_anomalies: counts before SMOTE
      – train_resampled_normals, train_resampled_anomalies: counts after SMOTE
      – test_normals, test_anomalies: counts in the test set
      – f1_score: F1 on the test set
      – tn, fp, fn, tp: confusion matrix entries
    """
    cols = [
        'M01_pct','M02_pct','M03_pct',
        'train_normals','train_anomalies',
        'train_resampled_normals','train_resampled_anomalies',
        'test_normals','test_anomalies',
        'f1_score','tn','fp','fn','tp', 'confusion_matrix', 'experiment_id'
    ]
    return pd.DataFrame(columns=cols)

def record_result(
    df,
    m01_pct, m02_pct, m03_pct,
    trainy, trainy_resampled,
    testy, f1, confusion_matrix,
    experiment_id=None
):
    tn, fp, fn, tp = confusion_matrix.ravel()

    train_normals  = sum(1 for y in trainy           if y == 0)
    train_anomalies = sum(1 for y in trainy           if y == 1)
    res_normals     = sum(1 for y in trainy_resampled if y == 0)
    res_anomalies   = sum(1 for y in trainy_resampled if y == 1)
    test_normals    = sum(1 for y in testy            if y == 0)
    test_anomalies  = sum(1 for y in testy            if y == 1)

    df.loc[len(df)] = [
        m01_pct, m02_pct, m03_pct,
        train_normals, train_anomalies,
        res_normals, res_anomalies,
        test_normals, test_anomalies,
        f1, tn, fp, fn, tp,
        confusion_matrix,
        experiment_id
    ]
    return df 
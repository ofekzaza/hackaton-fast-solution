#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../..')
import time
import pandas as pd
import numpy as np
from predict import Model

def evaluate(weights_path='weights.joblib'):
    t0 = time.time()
    model = Model()
    model.load(weights_path)
    val = pd.read_csv('../../dataset/public_validation_targets.csv')
    labels = pd.read_csv('../../dataset/private_labels.csv')
    val = val.merge(labels[['id', 'demand']], on='id', how='left')
    print(f'Loaded {len(val):,} rows in {time.time()-t0:.1f}s')

    t1 = time.time()
    preds = model.predict(val)
    y_true = val['demand'].values
    print(f'Predict: {time.time()-t1:.2f}s')
    print(f'Pred range: [{preds.min():.4f}, {preds.max():.4f}]')
    print(f'Pred mean: {preds.mean():.4f}  True mean: {y_true.mean():.4f}')
    print(f'Zero predictions: {(preds == 0).sum() / len(preds):.1%}')

    mae = np.abs(preds - y_true).mean()
    print(f'MAE: {mae:.4f}')

    for c in sorted(val['city'].unique()):
        mask = val['city'] == c
        mae_c = np.abs(preds[mask] - y_true[mask]).mean()
        print(f'  city {c}: MAE={mae_c:.4f}, n={mask.sum():,}  true_mean={y_true[mask].mean():.3f} pred_mean={preds[mask].mean():.3f}')

    zero_mask = y_true == 0
    nonzero_mask = ~zero_mask
    print(f'\nZero rows ({zero_mask.mean():.1%}): MAE={np.abs(preds[zero_mask]-y_true[zero_mask]).mean():.4f} pred_mean={preds[zero_mask].mean():.4f} zero_correct={(preds[zero_mask]==0).mean():.1%}')
    print(f'Nonzero rows: MAE={np.abs(preds[nonzero_mask]-y_true[nonzero_mask]).mean():.4f} pred_mean={preds[nonzero_mask].mean():.4f}')
    print(f'Total: {time.time()-t0:.1f}s')

if __name__ == '__main__':
    evaluate()

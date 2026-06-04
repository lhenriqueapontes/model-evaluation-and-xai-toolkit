from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_demo_model(seed=42):
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=seed, stratify=data.target)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return y_test, proba


def evaluate_threshold(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
        'f1': f1_score(y_true, pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, proba),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def threshold_table(y_true, proba):
    rows = [evaluate_threshold(y_true, proba, t) for t in np.arange(0.1, 1.0, 0.1)]
    return pd.DataFrame(rows).round(4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='reports')
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    y_true, proba = train_demo_model()
    threshold_table(y_true, proba).to_csv(out / 'threshold_metrics.csv', index=False)
    print(out / 'threshold_metrics.csv')


if __name__ == '__main__':
    main()

"""
Trains the fraud-scoring model.

IMPORTANT — read this before claiming any number from this script in an
interview:

The original reference version of this idea generated fraud rows with
`hours_since_sim_swap > 0` for every single fraud case and `-1` for every
single legitimate case. That means the label is directly encoded in one
input feature — the model doesn't learn a pattern, it learns "if this
column is not -1, output fraud." That's why it scored AUC = 1.0: perfect
scores on synthetic data are a symptom of data leakage, not a good model.
An interviewer who has trained any model at all will ask about that number
immediately.

This version fixes it two ways:
  1. A minority of LEGITIMATE users also have a past (harmless) SIM swap —
     e.g. they genuinely upgraded their SIM — so the feature is a strong
     signal, not a perfect one.
  2. The label is sampled from a probability (a logistic function of the
     weighted features + random noise), not assigned deterministically.
     This is what makes it behave like a real classification problem —
     the model has to actually learn feature weights, and some noise is
     irreducible, same as in a real fraud dataset.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
OUT_DIR = os.path.dirname(__file__)

FEATURES = [
    "hours_since_sim_swap", "is_known_device", "transaction_amount",
    "is_external_ip", "login_hour", "failed_attempts_1h",
    "account_age_days", "txns_last_24h",
]


def generate_data(n=10000):
    records = []
    for _ in range(n):
        # 12% of ALL users (legit or not) have some SIM-swap history.
        had_swap = np.random.rand() < 0.12
        if had_swap:
            # Could be recent (suspicious) or old/harmless (genuine carrier swap)
            hours = np.random.choice(
                [np.random.uniform(0, 2), np.random.uniform(2, 48), np.random.uniform(100, 3000)],
                p=[0.45, 0.30, 0.25],
            )
        else:
            hours = -1

        is_known_device = np.random.choice([1, 0], p=[0.75, 0.25])
        is_external_ip = np.random.choice([1, 0], p=[0.25, 0.75])
        login_hour = np.random.randint(0, 24)
        failed_attempts = np.random.choice([0, 1, 2, 3], p=[0.80, 0.12, 0.05, 0.03])
        account_age_days = np.random.randint(1, 1500)
        txns_last_24h = np.random.poisson(1.5)
        amount = float(np.random.exponential(3000))

        # Weighted risk signal -> probability of being fraud, then sample.
        # This is what a real logistic-regression-style scoring intuition
        # looks like: weighted sum of evidence, squashed to [0,1], plus noise.
        z = (
            -2.0
            + (3.5 if 0 <= hours < 2 else 1.8 if 0 <= hours < 48 else 0.0)
            + (1.2 * (1 - is_known_device))
            + (0.6 * is_external_ip)
            + (0.35 * failed_attempts)
            + (0.0008 * amount if amount > 50000 else 0)
            - (0.001 * min(account_age_days, 1000))  # older accounts = less risky
        )
        prob_fraud = 1 / (1 + np.exp(-z))
        label = int(np.random.rand() < prob_fraud)

        records.append({
            "hours_since_sim_swap": hours,
            "is_known_device": is_known_device,
            "transaction_amount": amount,
            "is_external_ip": is_external_ip,
            "login_hour": login_hour,
            "failed_attempts_1h": failed_attempts,
            "account_age_days": account_age_days,
            "txns_last_24h": txns_last_24h,
            "label": label,
        })

    df = pd.DataFrame(records).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


def train():
    print("Training SimGuard fraud model...")
    df = generate_data()
    n_fraud = int(df.label.sum())
    n_legit = int((df.label == 0).sum())
    fraud_pct = round(100 * n_fraud / len(df), 2)
    legit_pct = round(100 * n_legit / len(df), 2)
    print(f"Dataset: {len(df)} rows | fraud={n_fraud} ({fraud_pct}%) | legit={n_legit} ({legit_pct}%)")

    X, y = df[FEATURES], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED,
    )

    # 5-fold cross-validation on the training set — this is the number to
    # quote as "how I validated the model", not just a single test split.
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="roc_auc")
    print(f"5-fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nTest set — AUC: {auc:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    importances = dict(zip(FEATURES, model.feature_importances_.round(4)))
    print("Feature importances:", json.dumps(importances, indent=2))

    joblib.dump(model, os.path.join(OUT_DIR, "fraud_model.pkl"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))

    meta = {
        "features": FEATURES,
        "cv_auc_mean": round(cv_scores.mean(), 4),
        "cv_auc_std": round(cv_scores.std(), 4),
        "test_auc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "feature_importances": importances,
        "class_balance": {
            "fraud_count": n_fraud, "legit_count": n_legit,
            "fraud_pct": fraud_pct, "legit_pct": legit_pct,
        },
        "trained_on": f"{len(df)} synthetic samples",
        "model": "RandomForestClassifier",
    }
    with open(os.path.join(OUT_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved fraud_model.pkl, scaler.pkl, model_meta.json")


if __name__ == "__main__":
    train()

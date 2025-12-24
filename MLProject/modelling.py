"""
modelling.py
============
Script training Random Forest classifier for Credit Card Fraud Detection.

Author: I Gede Abhijana Prayata Wistara
Dicoding Username: wistaraocca
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import mlflow
import mlflow.sklearn
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Train Credit Card Fraud Detection Model')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees in the forest')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='Maximum depth of the tree')
    parser.add_argument('--min_samples_split', type=int, default=2,
                        help='Minimum samples required to split a node')
    parser.add_argument('--min_samples_leaf', type=int, default=1,
                        help='Minimum samples required at a leaf node')
    return parser.parse_args()


def load_data(train_path: str, test_path: str):

    print("=" * 67)
    print("Loading Data")
    print("=" * 67)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, args):

    print("\n" + "=" * 67)
    print("Training Model with MLflow")
    print("=" * 67)
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        print("\nLogging parameters.")
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("random_state", 42)

        print("Training Random Forest Classifier.")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print("Logging metrics.")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        mlflow.log_metric("true_negatives", tn)
        mlflow.log_metric("false_positives", fp)
        mlflow.log_metric("false_negatives", fn)
        mlflow.log_metric("true_positives", tp)

        print("\n" + "-" * 40)
        print("Model Performance Metrics:")
        print("-" * 40)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")

        print("\nLogging model...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="CreditCardFraudDetector-CI"
        )

        os.makedirs("model_output", exist_ok=True)
        model_path = "model_output/model.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")

        mlflow.set_tag("author", "wistaraocca")
        mlflow.set_tag("pipeline", "CI")
        
    return model, run_id


def main():
    print("\n" + "=" * 67)
    print("CREDIT CARD FRAUD DETECTION - CI PIPELINE")
    print("MLflow Project Training")
    print("=" * 67)

    args = parse_args()
    print(f"\nHyperparameters:")
    print(f"  - n_estimators: {args.n_estimators}")
    print(f"  - max_depth: {args.max_depth}")
    print(f"  - min_samples_split: {args.min_samples_split}")
    print(f"  - min_samples_leaf: {args.min_samples_leaf}")

    X_train, X_test, y_train, y_test = load_data(
        train_path="../creditcard_preprocessing/creditcard_train.csv",
        test_path="../creditcard_preprocessing/creditcard_test.csv"
    )

    model, run_id = train_model(X_train, X_test, y_train, y_test, args)
    
    print("\n" + "=" * 67)
    print("TRAINING COMPLETED")
    print("=" * 67)
    print(f"Run ID: {run_id}")

    with open("latest_run_id.txt", "w") as f:
        f.write(run_id)
    print(f"Run ID saved to: latest_run_id.txt")


if __name__ == "__main__":
    main()

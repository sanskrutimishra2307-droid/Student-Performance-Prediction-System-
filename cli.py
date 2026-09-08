"""
Command-Line Interface (CLI) for Student Performance Prediction System.
Supports automated training, evaluation, benchmark generation, and batch predictions.
"""

import argparse
import sys
import os
import json
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.trainer import ModelTrainer
from src.explainability import ExplainabilityEngine
from src.config import DEFAULT_DATASET_PATH, SAMPLE_BATCH_PATH

def run_train(args):
    print(f"[+] Initializing Model Training with Feature Selection ({args.fs_method}, k={args.k})...")
    trainer = ModelTrainer(
        dataset_path=args.data,
        target_scheme=args.target,
        feature_selection_method=args.fs_method,
        n_features=args.k,
        scaler_type=args.scaler
    )
    results = trainer.run_full_pipeline(
        tune_hyperparameters=args.tune,
        save_artifacts=True
    )
    print("\n[SUCCESS] Training Complete!")
    print(f"[*] Best Model: {results['best_model_name']}")
    print(f"[*] Best Test F1-Score: {results['best_score_f1']:.4f}")
    print(f"[*] Top Selected Features: {results['selected_features']}")

def run_predict(args):
    print("[*] Running Single Student Prediction...")
    student_data = {
        "age": args.age,
        "gender": args.gender,
        "school_type": args.school,
        "parent_education": args.parent_edu,
        "study_hours": args.study_hours,
        "attendance_percentage": args.attendance,
        "internet_access": args.internet,
        "travel_time": args.travel,
        "extra_activities": args.activities,
        "study_method": args.method,
        "math_score": args.math,
        "science_score": args.science,
        "english_score": args.english
    }
    res = ModelTrainer.predict_single(student_data)
    xai = ExplainabilityEngine.analyze_student_risk_factors(student_data, res["prediction"])
    
    print("\n--- Prediction Output ---")
    print(f"Predicted Performance: {res['prediction']}")
    print(f"Confidence: {res['confidence']:.2%}")
    print(f"Class Probabilities: {res['probabilities']}")
    print(f"Urgency Level: {xai['urgency']}")
    print(f"Recommendations: {xai['recommended_interventions']}")

def run_batch(args):
    print(f"[*] Running Batch Predictions on '{args.file}'...")
    df = pd.read_csv(args.file)
    results_df = ModelTrainer.predict_batch(df)
    
    out_path = args.out or "student_batch_predictions.csv"
    results_df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Batch results saved to '{out_path}' ({len(results_df)} records)")

def main():
    parser = argparse.ArgumentParser(description="Student Performance Prediction & Feature Selection Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train subparser
    train_parser = subparsers.add_parser("train", help="Train pipeline models")
    train_parser.add_argument("--data", type=str, default=str(DEFAULT_DATASET_PATH), help="Path to CSV dataset")
    train_parser.add_argument("--target", type=str, default="3-tier", choices=["3-tier", "binary", "multi-grade"], help="Target scheme")
    train_parser.add_argument("--fs_method", type=str, default="consensus", help="Feature selection method")
    train_parser.add_argument("--k", type=int, default=8, help="Number of features to select")
    train_parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "robust", "minmax"])
    train_parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    
    # Predict subparser
    pred_parser = subparsers.add_parser("predict", help="Predict for a single student")
    pred_parser.add_argument("--age", type=int, default=16)
    pred_parser.add_argument("--gender", type=str, default="female")
    pred_parser.add_argument("--school", type=str, default="public")
    pred_parser.add_argument("--parent_edu", type=str, default="graduate")
    pred_parser.add_argument("--study_hours", type=float, default=4.5)
    pred_parser.add_argument("--attendance", type=float, default=85.0)
    pred_parser.add_argument("--internet", type=str, default="yes")
    pred_parser.add_argument("--travel", type=str, default="<15 min")
    pred_parser.add_argument("--activities", type=str, default="yes")
    pred_parser.add_argument("--method", type=str, default="notes")
    pred_parser.add_argument("--math", type=float, default=70.0)
    pred_parser.add_argument("--science", type=float, default=72.0)
    pred_parser.add_argument("--english", type=float, default=75.0)
    
    # Batch subparser
    batch_parser = subparsers.add_parser("batch", help="Batch predict on CSV")
    batch_parser.add_argument("--file", type=str, default=str(SAMPLE_BATCH_PATH), help="Path to input CSV")
    batch_parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    
    args = parser.parse_args()
    
    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "batch":
        run_batch(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

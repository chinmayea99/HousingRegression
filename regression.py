import pandas as pd
import numpy as np
from utils import load_data, preprocess_data, train_models, evaluate_models
import json

def main():
    print("Loading Boston Housing dataset...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    print("\nTraining models...")
    models = train_models(X_train, y_train)
    
    print("\nEvaluating models...")
    results = evaluate_models(models, X_test, y_test)
    
    print("\n=== REGRESSION RESULTS ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  MSE: {metrics['MSE']:.4f}")
        print(f"  R²:  {metrics['R2']:.4f}")
    
    # Save results to JSON
    with open('regression_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to regression_results.json")

if __name__ == "__main__":
    main()

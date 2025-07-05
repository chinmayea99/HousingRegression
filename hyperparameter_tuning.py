import pandas as pd
import numpy as np
from utils import load_data, preprocess_data, train_models_with_hyperparameters, evaluate_models
import json

def main():
    print("Loading Boston Housing dataset...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    print("\nTraining models with hyperparameter tuning...")
    best_models, best_params = train_models_with_hyperparameters(X_train, y_train)
    
    print("\nEvaluating tuned models...")
    results = evaluate_models(best_models, X_test, y_test)
    
    print("\n=== HYPERPARAMETER TUNING RESULTS ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Best Parameters: {best_params[model_name]}")
        print(f"  MSE: {metrics['MSE']:.4f}")
        print(f"  R²:  {metrics['R2']:.4f}")
    
    # Save results
    final_results = {
        'model_performance': results,
        'best_parameters': best_params
    }
    
    with open('hyperparameter_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\nResults saved to hyperparameter_results.json")

if __name__ == "__main__":
    main()

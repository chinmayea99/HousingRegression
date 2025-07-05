import pandas as pd
import numpy as np
from utils import load_data, preprocess_data, train_models, train_models_with_hyperparameters, evaluate_models
import json

def create_performance_report():
    """Create comprehensive performance comparison report"""
    
    print("=== HOUSING REGRESSION PERFORMANCE COMPARISON REPORT ===\n")
    
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    print("Dataset Information:")
    print(f"- Total samples: {len(df)}")
    print(f"- Features: {len(df.columns)-1}")
    print(f"- Training samples: {len(X_train)}")
    print(f"- Test samples: {len(X_test)}")
    print(f"- Target variable: MEDV (Median Home Value)")
    print()
    
    # Basic models
    print("1. BASIC REGRESSION MODELS (Without Hyperparameter Tuning)")
    print("=" * 60)
    basic_models = train_models(X_train, y_train)
    basic_results = evaluate_models(basic_models, X_test, y_test)
    
    for model_name, metrics in basic_results.items():
        print(f"{model_name}:")
        print(f"  MSE: {metrics['MSE']:.4f}")
        print(f"  R²:  {metrics['R2']:.4f}")
        print()
    
    # Tuned models
    print("2. HYPERPARAMETER TUNED MODELS")
    print("=" * 60)
    tuned_models, best_params = train_models_with_hyperparameters(X_train, y_train)
    tuned_results = evaluate_models(tuned_models, X_test, y_test)
    
    for model_name, metrics in tuned_results.items():
        print(f"{model_name}:")
        print(f"  Best Parameters: {best_params[model_name]}")
        print(f"  MSE: {metrics['MSE']:.4f}")
        print(f"  R²:  {metrics['R2']:.4f}")
        print()
    
    # Comparison
    print("3. PERFORMANCE IMPROVEMENT COMPARISON")
    print("=" * 60)
    
    comparison_data = []
    for model_name in basic_results.keys():
        basic_mse = basic_results[model_name]['MSE']
        basic_r2 = basic_results[model_name]['R2']
        tuned_mse = tuned_results[model_name]['MSE']
        tuned_r2 = tuned_results[model_name]['R2']
        
        mse_improvement = ((basic_mse - tuned_mse) / basic_mse) * 100
        r2_improvement = ((tuned_r2 - basic_r2) / basic_r2) * 100
        
        comparison_data.append({
            'Model': model_name,
            'Basic_MSE': basic_mse,
            'Tuned_MSE': tuned_mse,
            'Basic_R2': basic_r2,
            'Tuned_R2': tuned_r2,
            'MSE_Improvement_%': mse_improvement,
            'R2_Improvement_%': r2_improvement
        })
        
        print(f"{model_name}:")
        print(f"  MSE Improvement: {mse_improvement:.2f}%")
        print(f"  R² Improvement: {r2_improvement:.2f}%")
        print()
    
    # Save comprehensive results
    final_report = {
        'dataset_info': {
            'total_samples': len(df),
            'features': len(df.columns)-1,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        },
        'basic_models': basic_results,
        'tuned_models': tuned_results,
        'best_parameters': best_params,
        'comparison': comparison_data
    }
    
    with open('final_performance_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print("4. BEST PERFORMING MODEL")
    print("=" * 60)
    best_model = min(tuned_results.items(), key=lambda x: x[1]['MSE'])
    print(f"Best Model: {best_model[0]}")
    print(f"MSE: {best_model[1]['MSE']:.4f}")
    print(f"R²: {best_model[1]['R2']:.4f}")
    print()
    
    print("Report saved to 'final_performance_report.json'")

if __name__ == "__main__":
    create_performance_report()

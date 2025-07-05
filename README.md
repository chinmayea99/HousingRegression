# Housing Regression MLOps Assignment
# MLOps Housing Regression Assignment - Final Report

## Student Information
- **Name:** Chinmaye A
- **Assignment:** MLOps Pipeline for Housing Price Prediction
- **Date:** July 5, 2025

## Executive Summary
This project implements a complete MLOps pipeline for predicting house prices using the Boston Housing dataset. The pipeline includes automated model training, hyperparameter tuning, performance comparison, and CI/CD automation using GitHub Actions.

## 1. Problem Statement & Objectives

### Problem Statement
Design, implement, and automate a complete machine learning workflow to predict house prices using classical machine learning models with the Boston Housing dataset.

### Key Objectives
- Implement minimum 3 regression models
- Compare performance using MSE and R² metrics
- Implement hyperparameter tuning (minimum 3 parameters per model)
- Create automated CI/CD pipeline using GitHub Actions
- Follow proper Git branching strategy
- Maintain comprehensive documentation

## 2. Dataset Description

### Boston Housing Dataset
- **Source:** http://lib.stat.cmu.edu/datasets/boston
- **Total Samples:** 506
- **Features:** 13 input features + 1 target variable
- **Target Variable:** MEDV (Median Home Value in $1000s)

### Feature Description
| Feature | Description |
|---------|-------------|
| CRIM | Per capita crime rate by town |
| ZN | Proportion of residential land zoned for lots over 25,000 sq.ft |
| INDUS | Proportion of non-retail business acres per town |
| CHAS | Charles River dummy variable (1 if tract bounds river; 0 otherwise) |
| NOX | Nitric oxides concentration (parts per 10 million) |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of owner-occupied units built prior to 1940 |
| DIS | Weighted distances to five Boston employment centres |
| RAD | Index of accessibility to radial highways |
| TAX | Full-value property-tax rate per $10,000 |
| PTRATIO | Pupil-teacher ratio by town |
| B | Proportion of blacks by town |
| LSTAT | % lower status of the population |
| MEDV | Median value of owner-occupied homes in $1000's (TARGET) |

## 3. Repository Structure & Branch Strategy

### Repository Structure
```
HousingRegression/
├── .github/workflows/
│   └── ci.yml                 # GitHub Actions CI/CD pipeline
├── utils.py                   # Utility functions for data processing and modeling
├── regression.py              # Basic regression models script
├── hyperparameter_tuning.py   # Hyperparameter tuning script
├── create_report.py           # Performance comparison report generator
├── requirements.txt           # Python dependencies
└── README.md                  # Comprehensive documentation
```

### Branch Strategy
1. **main branch:** Contains final merged code with complete pipeline
2. **reg branch:** Contains basic regression models without hyperparameter tuning
3. **hyper branch:** Contains hyperparameter tuning functionality

All branches are preserved as per assignment requirements.

## 4. Models Implemented

### 4.1 Linear Regression
- **Algorithm:** Ordinary Least Squares
- **Hyperparameters Tuned:**
  - `fit_intercept`: [True, False]
  - `positive`: [True, False]
  - `copy_X`: [True, False]
- **Use Case:** Baseline model for linear relationships

### 4.2 Random Forest Regressor
- **Algorithm:** Ensemble of Decision Trees
- **Hyperparameters Tuned:**
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [None, 10, 20]
  - `min_samples_split`: [2, 5, 10]
- **Use Case:** Handles non-linear relationships and feature interactions

### 4.3 Support Vector Regressor (SVR)
- **Algorithm:** Support Vector Machine for regression
- **Hyperparameters Tuned:**
  - `C`: [0.1, 1, 10]
  - `gamma`: ['scale', 'auto', 0.1]
  - `kernel`: ['rbf', 'linear', 'poly']
- **Use Case:** Effective for high-dimensional data

## 5. Methodology

### 5.1 Data Preprocessing
1. **Data Loading:** Custom function to load Boston Housing dataset from original source
2. **Train-Test Split:** 80-20 split with random_state=42 for reproducibility
3. **Feature Scaling:** StandardScaler applied to normalize feature ranges
4. **Data Validation:** Shape and structure verification

### 5.2 Model Training Pipeline
1. **Basic Models:** Train all three models with default parameters
2. **Hyperparameter Tuning:** Use GridSearchCV with 5-fold cross-validation
3. **Model Evaluation:** Calculate MSE and R² on test set
4. **Performance Comparison:** Compare basic vs. tuned models

### 5.3 Evaluation Metrics
- **MSE (Mean Squared Error):** Measures average squared differences between actual and predicted values
- **R² (R-squared):** Measures proportion of variance explained by the model (0-1, higher is better)

## 6. Performance Results

### 6.1 Basic Models (Without Hyperparameter Tuning)

| Model | MSE | R² | Performance |
|-------|-----|----|-----------  |
| Linear Regression | ~21.52 | ~0.67 | Baseline performance |
| Random Forest | ~11.48 | ~0.82 | Best among basic models |
| SVR | ~43.89 | ~0.33 | Needs tuning |

### 6.2 Hyperparameter Tuned Models

| Model | MSE | R² | Best Parameters | Improvement |
|-------|-----|----|--------------  |-------------|
| Linear Regression | ~21.52 | ~0.67 | fit_intercept=True, positive=False | Minimal (linear model) |
| Random Forest | ~8.94 | ~0.86 | n_estimators=200, max_depth=20 | ~22% MSE improvement |
| SVR | ~18.45 | ~0.72 | C=10, gamma=0.1, kernel='rbf' | ~58% MSE improvement |

### 6.3 Performance Comparison Summary

**Best Performing Model:** Random Forest Regressor (Tuned)
- **MSE:** 8.94
- **R²:** 0.86
- **Key Strengths:** Handles non-linear relationships, robust to outliers

**Most Improved Model:** Support Vector Regressor
- **MSE Improvement:** 58%
- **R² Improvement:** 118%
- **Key Insight:** SVR significantly benefits from hyperparameter tuning

## 7. MLOps Implementation

### 7.1 GitHub Actions CI/CD Pipeline
```yaml
name: CI Pipeline
on:
  push:
    branches: [ main, reg, hyper ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - Checkout code
    - Set up Python 3.9
    - Install dependencies
    - Run regression analysis
    - Run hyperparameter tuning
    - Upload results as artifacts
```

### 7.2 Automated Workflow Features
- **Trigger:** Automatic execution on push to any branch
- **Environment:** Ubuntu latest with Python 3.9
- **Dependencies:** Automated installation from requirements.txt
- **Execution:** Runs both basic and hyperparameter tuning scripts
- **Artifacts:** Saves results as downloadable artifacts

### 7.3 Modular Code Structure
- **utils.py:** Core functions for data loading, preprocessing, training, and evaluation
- **regression.py:** Script for basic model training and evaluation
- **hyperparameter_tuning.py:** Script for hyperparameter optimization
- **create_report.py:** Comprehensive performance comparison generator

## 8. Key Technical Implementations

### 8.1 Data Loading Function
```python
def load_data():
    """Load Boston Housing dataset from original source"""
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    # Process and structure data
    return df
```

### 8.2 Hyperparameter Tuning Implementation
```python
def train_models_with_hyperparameters(X_train, y_train):
    """Train models with GridSearchCV optimization"""
    grid_search = GridSearchCV(
        model, param_grid, cv=5, 
        scoring='neg_mean_squared_error', n_jobs=-1
    )
    return best_models, best_params
```

### 8.3 Performance Evaluation
```python
def evaluate_models(models, X_test, y_test):
    """Calculate MSE and R² for all models"""
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    return results
```

## 9. Lessons Learned & Best Practices

### 9.1 MLOps Best Practices Implemented
1. **Version Control:** Proper Git branching strategy with preserved branches
2. **Reproducibility:** Fixed random seeds and documented dependencies
3. **Automation:** CI/CD pipeline for continuous integration
4. **Documentation:** Comprehensive README and inline code documentation
5. **Modularity:** Separated concerns into different modules and functions

### 9.2 Key Insights
1. **Hyperparameter Tuning Impact:** Significant performance improvements, especially for SVR
2. **Model Selection:** Random Forest performs best for this dataset
3. **Feature Scaling:** Critical for SVR performance
4. **Cross-Validation:** Essential for reliable hyperparameter selection

## 10. Conclusion

This project successfully demonstrates a complete MLOps pipeline for housing price prediction with the following achievements:

### ✅ Requirements Met
- [x] Implemented 3 regression models (Linear Regression, Random Forest, SVR)
- [x] Performance comparison using MSE and R² metrics
- [x] Hyperparameter tuning with minimum 3 parameters per model
- [x] Proper Git branching strategy (main, reg, hyper)
- [x] GitHub Actions CI/CD automation
- [x] Modular code structure with utils.py
- [x] Comprehensive documentation
- [x] All branches preserved

### 🎯 Key Outcomes
- **Best Model:** Random Forest Regressor (R² = 0.86, MSE = 8.94)
- **Biggest Improvement:** SVR with 58% MSE reduction through tuning
- **Automation:** Full CI/CD pipeline with artifact generation
- **Documentation:** Complete technical documentation and performance reports

### 📈 Performance Summary
The Random Forest Regressor achieved the best performance with an R² score of 0.86, explaining 86% of the variance in housing prices. The hyperparameter tuning process significantly improved model performance, particularly for the SVR model which saw a 58% reduction in MSE.

This project demonstrates proficiency in MLOps practices including automated pipelines, version control, performance monitoring, and comprehensive documentation - essential skills for production machine learning systems.

---

## Appendix

### A. Repository Links
- **GitHub Repository:** (https://github.com/chinmayea99/HousingRegression/tree/main)
- **Main Branch:** Contains final merged code
- **Reg Branch:** Basic regression implementation
- **Hyper Branch:** Hyperparameter tuning implementation

### B. Generated Files
- `regression_results.json`: Basic model performance results
- `hyperparameter_results.json`: Tuned model results with best parameters
- `final_performance_report.json`: Comprehensive comparison report

### C. Dependencies
```
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.2.0
```

### D. Execution Commands
```bash
# Basic regression
python regression.py

# Hyperparameter tuning
python hyperparameter_tuning.py

# Performance comparison
python create_report.py
```

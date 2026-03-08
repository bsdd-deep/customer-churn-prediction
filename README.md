# Customer Churn Prediction

Predicting customer churn using machine learning classification models.

## Overview
This project uses historical customer data to build predictive models that identify customers likely to churn. Multiple algorithms are compared and the best model is selected based on performance metrics.

## Dataset
- Source: Telecom customer churn dataset
- Features: 20+ customer attributes (demographics, services, contracts)
- Target: Binary classification (Churn/No Churn)
- Size: ~7000 samples

## Models Implemented
1. **Logistic Regression** - Baseline model
2. **Random Forest** - Ensemble method
3. **Gradient Boosting** - Advanced ensemble
4. **SVM** - Support Vector Machine

## Results
- Best Model: Gradient Boosting
- Accuracy: 87.3%
- Precision: 0.85
- Recall: 0.82
- F1-Score: 0.84

## Usage
```python
python train.py --model gbm --test-split 0.2
python predict.py --input customers.csv --model models/gbm_final.pkl
```

## Requirements
- Python 3.8+
- scikit-learn>=1.0
- pandas>=1.3
- numpy>=1.21
- matplotlib>=3.4

## Installation
```bash
pip install -r requirements.txt
```

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import argparse

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess(df):
    # Handle categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Scale features
    scaler = StandardScaler()
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_model(X_train, y_train):
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/churn.csv")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--output", default="models/gbm_final.pkl")
    args = parser.parse_args()
    
    df = load_data(args.data)
    X, y, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_split)
    
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)
    
    with open(args.output, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {args.output}")

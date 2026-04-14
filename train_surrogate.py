import sqlite3
import pandas as pd
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys

# Add path to load extract_features
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from alpha_ranker import extract_features

# Configuration
DB_PATH = "alpha_results.db"
MODEL_DIR = os.path.join("data", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_ranker.pkl")

def get_training_data():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database {DB_PATH} not found.")
        return None
        
    conn = sqlite3.connect(DB_PATH)
    # Target definition: High Sharpe (> 1.2) and High Fitness (> 1.0)
    query = """
    SELECT expression, sharpe, fitness
    FROM alphas
    WHERE sharpe IS NOT NULL AND sharpe != 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) < 100:
        print(f"Warning: Only {len(df)} records found. Need more data for robust training.")
        
    return df

def train_surrogate():
    df = get_training_data()
    if df is None or len(df) < 10:
        print("Not enough data to train surrogate model.")
        return

    print(f"Loaded {len(df)} samples from database.")
    
    # Define Target (1 for Good, 0 for Bad)
    # We set a threshold: Sharpe >= 1.0
    df['is_elite'] = (df['sharpe'] >= 1.0).astype(int)
    
    print("Class Distribution:")
    print(df['is_elite'].value_counts())
    
    # Feature Extraction
    features_list = []
    for expr in df['expression']:
        feats = extract_features(expr)
        features_list.append(feats)
        
    X = pd.DataFrame(features_list)
    y = df['is_elite']
    
    # If all classes are the same, we can't train
    if len(y.unique()) < 2:
        print("Error: Target variable has only one class. Cannot train XGBoost.")
        return
        
    # Standardize/Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Model
    print("Training XGBoost Surrogate Model...")
    # Use robust params to deal with imbalanced data
    scale_pos_weight = sum(y == 0) / max(1, sum(y == 1))
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    
    # Save Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}. alpha_ranker.py will now use it automatically.")

if __name__ == "__main__":
    train_surrogate()

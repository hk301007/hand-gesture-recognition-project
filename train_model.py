import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

def train():
    # Force creation of models directory
    os.makedirs("models", exist_ok=True)
    
    print("Loading training data from 'data/raw/session1_train.csv'...")
    train_df = pd.read_csv("data/raw/session1_train.csv")
    
    # Feature columns
    raw_cols = [f"raw_{i}" for i in range(63)]
    inv_cols = [f"inv_{i}" for i in range(8)]
    y_train = train_df["label"]
    
    # 1. Train & Save Raw Coordinates Model
    print("Training Raw Coordinates Model...")
    model_raw = RandomForestClassifier(n_estimators=100, random_state=42)
    model_raw.fit(train_df[raw_cols], y_train)
    joblib.dump(model_raw, "models/model_raw.pkl")
    
    # 2. Train & Save Invariant Features Model
    print("Training Invariant Features Model...")
    model_inv = RandomForestClassifier(n_estimators=100, random_state=42)
    model_inv.fit(train_df[inv_cols], y_train)
    joblib.dump(model_inv, "models/gesture_model_invariant.pkl")
    
    print("\nSUCCESS: Both models saved in the 'models/' directory:")
    print("  - models/model_raw.pkl")
    print("  - models/gesture_model_invariant.pkl")

if __name__ == "__main__":
    train()
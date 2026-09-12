import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def evaluate():
    # Load Datasets
    train_df = pd.read_csv("data/raw/session1_train.csv")
    test_cross_df = pd.read_csv("data/raw/session2_test.csv")
    
    # Split session 1 into same-session train/test
    _, test_same_df = train_test_split(
        train_df, test_size=0.3, random_state=42, stratify=train_df["label"]
    )
    
    # Load Trained Models
    model_raw = joblib.load("models/model_raw.pkl")
    model_inv = joblib.load("models/gesture_model_invariant.pkl")
    
    raw_cols = [f"raw_{i}" for i in range(63)]
    inv_cols = [f"inv_{i}" for i in range(8)]
    
    # Compute Accuracies
    acc_raw_same = accuracy_score(test_same_df["label"], model_raw.predict(test_same_df[raw_cols]))
    acc_raw_cross = accuracy_score(test_cross_df["label"], model_raw.predict(test_cross_df[raw_cols]))
    
    acc_inv_same = accuracy_score(test_same_df["label"], model_inv.predict(test_same_df[inv_cols]))
    acc_inv_cross = accuracy_score(test_cross_df["label"], model_inv.predict(test_cross_df[inv_cols]))
    
    # Build 2x2 Matrix
    matrix_df = pd.DataFrame({
        "Same-Session Test": [f"{acc_raw_same*100:.2f}%", f"{acc_inv_same*100:.2f}%"],
        "Cross-Session Test": [f"{acc_raw_cross*100:.2f}%", f"{acc_inv_cross*100:.2f}%"]
    }, index=["Raw Coordinates (63)", "Invariant Features (8)"])
    
    print("\n=== FOUR-CELL GENERALIZATION TABLE ===")
    print(matrix_df)
    print("=======================================\n")

if __name__ == "__main__":
    evaluate()
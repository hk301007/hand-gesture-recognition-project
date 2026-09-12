import time
import numpy as np
import pandas as pd
import joblib

def benchmark():
    model_inv = joblib.load("models/gesture_model_invariant.pkl")
    test_df = pd.read_csv("data/raw/session2_test.csv")
    inv_cols = [f"inv_{i}" for i in range(8)]
    sample_input = test_df[inv_cols].iloc[:1]
    
    # Measure Classifier Inference Latency
    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        _ = model_inv.predict(sample_input)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # milliseconds
        
    avg_latency = np.mean(latencies)
    print("\n=== LATENCY BENCHMARKING RESULTS ===")
    print(f"Classifier Inference Latency: {avg_latency:.4f} ms per frame")
    print("=====================================\n")

if __name__ == "__main__":
    benchmark()
# Real-Time Hand Gesture Recognition Pipeline

## Overview
This project implements an end-to-end Computer Vision pipeline to recognize hand gestures in real-time. Built using Python, OpenCV, MediaPipe, and scikit-learn, it extracts 21 hand landmarks (63 raw coordinates) and computes 8 scale- and translation-invariant geometric features. The pipeline evaluates operational generalization across distinct testing conditions and measures system latencies.

## Project Structure
```text
hand-gesture-recognition/
├── data/
│   └── raw/
│       ├── session1_train.csv
│       └── session2_test.csv
├── models/
│   ├── model_raw.pkl
│   └── gesture_model_invariant.pkl
├── benchmark.py
├── collect_data.py
├── evaluate.py
├── extract_features.py
├── realtime.py
├── requirements.txt
└── train_model.py
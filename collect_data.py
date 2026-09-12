import cv2
import mediapipe as mp
import pandas as pd
import os
from extract_features import extract_landmarks_and_features

try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_drawing

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def collect_gesture_data(output_csv, label_name, samples_target=150):
    cap = cv2.VideoCapture(0)
    records = []
    count = 0
    recording = False
    
    print(f"\n--- Collecting for Class: '{label_name}' ---")
    print("Press 's' to start recording. Press 'q' to quit early.")
    
    while cap.isOpened() and count < samples_target:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if recording:
                    raw_feats, inv_feats = extract_landmarks_and_features(hand_landmarks)
                    records.append(raw_feats + inv_feats + [label_name])
                    count += 1
        
        status_color = (0, 255, 0) if recording else (0, 0, 255)
        status_text = f"Recording: {count}/{samples_target}" if recording else "Press 's' to Start"
        cv2.putText(frame, f"Class: {label_name} | {status_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.imshow("Data Collector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            recording = True
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    raw_cols = [f"raw_{i}" for i in range(63)]
    inv_cols = [f"inv_{i}" for i in range(8)]
    df = pd.DataFrame(records, columns=raw_cols + inv_cols + ["label"])
    
    file_exists = os.path.isfile(output_csv)
    df.to_csv(output_csv, mode='a', header=not file_exists, index=False)
    print(f"Saved {count} samples for '{label_name}' to {output_csv}")

if __name__ == "__main__":
    session_choice = input("Enter session identifier ('train' or 'test'): ").strip().lower()
    target_csv = "data/raw/session1_train.csv" if session_choice == "train" else "data/raw/session2_test.csv"
    
    classes = ["open_hand", "fist", "peace", "thumbs_up"]
    for gesture_class in classes:
        collect_gesture_data(target_csv, gesture_class, samples_target=150)
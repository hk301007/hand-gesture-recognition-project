import numpy as np

def extract_landmarks_and_features(hand_landmarks):
    # 1. Extract 63 Raw Coordinates (x, y, z for 21 keypoints)
    raw_coords = []
    for lm in hand_landmarks.landmark:
        raw_coords.extend([lm.x, lm.y, lm.z])
    
    # 2. Extract 8 Invariant Geometric Features
    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
    mcp_middle = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z])
    
    # Scale normalization factor (wrist to middle finger MCP distance)
    scale = np.linalg.norm(mcp_middle - wrist) + 1e-6
    
    # Fingertip vectors
    thumb_tip  = np.array([hand_landmarks.landmark[4].x,  hand_landmarks.landmark[4].y,  hand_landmarks.landmark[4].z])
    index_tip  = np.array([hand_landmarks.landmark[8].x,  hand_landmarks.landmark[8].y,  hand_landmarks.landmark[8].z])
    middle_tip = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y, hand_landmarks.landmark[12].z])
    ring_tip   = np.array([hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y, hand_landmarks.landmark[16].z])
    pinky_tip  = np.array([hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y, hand_landmarks.landmark[20].z])
    
    inv_features = [
        np.linalg.norm(thumb_tip - wrist) / scale,
        np.linalg.norm(index_tip - wrist) / scale,
        np.linalg.norm(middle_tip - wrist) / scale,
        np.linalg.norm(ring_tip - wrist) / scale,
        np.linalg.norm(pinky_tip - wrist) / scale,
        np.linalg.norm(thumb_tip - index_tip) / scale,
        np.linalg.norm(index_tip - middle_tip) / scale,
        np.linalg.norm(middle_tip - pinky_tip) / scale
    ]
    
    return raw_coords, inv_features
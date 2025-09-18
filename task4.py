# gesture_system.py
"""
Hand Gesture Recognition System - single-file all-in-one.

Features:
- Landmark extraction using MediaPipe (extract_landmarks)
- Landmark-based classifier (RandomForest or MLP) (train_landmark)
- Image-based transfer learning model using MobileNetV2 (train_image)
- Realtime webcam demos: landmark-based (fast) and image-based (slower) (realtime_landmark, realtime_image)
- Simple evaluation and reporting

Usage examples:
  python gesture_system.py extract_landmarks --input_dir data_images --out_csv hand_landmarks.csv
  python gesture_system.py train_landmark --csv hand_landmarks.csv --out model_landmark.joblib
  python gesture_system.py realtime_landmark --model model_landmark.joblib
  python gesture_system.py train_image --data_dir data_images --out model_image.h5
  python gesture_system.py realtime_image --model model_image.h5

Notes:
- For image training, data_dir should contain train/, val/, test/ folders with subfolders per class.
- For landmark extraction, input_dir should contain subfolders per class with images.
"""

import argparse
import os
import sys
import time
import csv
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

# try imports and show useful message if missing
try:
    import mediapipe as mp
except Exception as e:
    print("Missing mediapipe. Install it with: pip install mediapipe")
    raise e
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    import joblib
except Exception as e:
    print("Missing scikit-learn or joblib. Install with: pip install scikit-learn joblib")
    raise e
# TensorFlow import optional (only required for image model)
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing import image_dataset_from_directory
except Exception:
    tf = None

# ---- Utility functions ----

def list_class_folders(path: str) -> List[str]:
    """Return sorted class folder names in a directory."""
    names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    names.sort()
    return names

# ---- Landmark extraction with MediaPipe ----

def extract_landmarks_from_image(image_bgr, hands_detector):
    """Return flattened landmark vector [x,y,z,...] for first detected hand, or None."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(img_rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0]
    flat = []
    for p in lm.landmark:
        flat.extend([p.x, p.y, p.z])   # normalized coordinates (0..1)
    return flat

def extract_landmarks_folder(input_dir: str, out_csv: str, verbose=True):
    """
    Walks subfolders in input_dir (one subfolder per class),
    processes images, and writes a CSV with header: label, f0, f1, ...
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    classes = list_class_folders(input_dir)
    rows = []
    num_skipped =

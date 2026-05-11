"""
Advanced Sign Language Detection System
========================================
Features:
- Real-time ASL alphabet & word detection
- MediaPipe hand landmark tracking (21 points per hand)
- Gesture confidence scoring
- Sentence builder with word history
- FPS counter & performance stats
- Screenshot & recording support
- Two-hand detection
- Dynamic gesture (motion-based) detection
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os
import json
from collections import deque, Counter
from datetime import datetime

from utils.hand_tracker import HandTracker
from utils.gesture_classifier import GestureClassifier
from utils.sentence_builder import SentenceBuilder
from utils.visualizer import Visualizer
from utils.recorder import Recorder


def main():
    # ── Configuration ──────────────────────────────────────────────
    CONFIG = {
        "camera_index": 0,
        "width": 1280,
        "height": 720,
        "confidence_threshold": 0.75,
        "prediction_smoothing": 10,   # frames to average
        "sentence_pause_sec": 2.0,    # silence → new word
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.5,
    }

    # ── Initialize modules ──────────────────────────────────────────
    tracker    = HandTracker(CONFIG)
    classifier = GestureClassifier(CONFIG["confidence_threshold"])
    builder    = SentenceBuilder(pause_threshold=CONFIG["sentence_pause_sec"])
    visualizer = Visualizer()
    recorder   = Recorder(output_dir="recordings")

    # ── Camera setup ────────────────────────────────────────────────
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["height"])
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check camera_index in CONFIG.")
        return

    # ── State ───────────────────────────────────────────────────────
    prediction_buffer = deque(maxlen=CONFIG["prediction_smoothing"])
    fps_times         = deque(maxlen=30)
    is_recording      = False
    show_landmarks    = True
    show_stats        = True
    frame_count       = 0

    print("\n🤟 Sign Language Detection — Running")
    print("━" * 45)
    print("  [Q]     Quit")
    print("  [R]     Start / Stop recording")
    print("  [S]     Screenshot")
    print("  [L]     Toggle landmarks")
    print("  [I]     Toggle stats panel")
    print("  [C]     Clear sentence")
    print("  [SPACE] Add space to sentence")
    print("━" * 45 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame grab failed.")
            break

        frame = cv2.flip(frame, 1)           # mirror for natural feel
        frame_count += 1

        # ── FPS tracking ─────────────────────────────────────────────
        fps_times.append(time.time())
        fps = len(fps_times) / (fps_times[-1] - fps_times[0] + 1e-6) if len(fps_times) > 1 else 0

        # ── Hand detection + landmark extraction ─────────────────────
        results, landmarks_list = tracker.process(frame)

        current_prediction  = None
        current_confidence  = 0.0
        smoothed_prediction = None

        if landmarks_list:
            # Classify dominant (most-confident) hand
            for hand_data in landmarks_list:
                prediction, confidence = classifier.classify(hand_data["landmarks"])
                if confidence > current_confidence:
                    current_prediction = prediction
                    current_confidence = confidence

            if current_prediction:
                prediction_buffer.append(current_prediction)

            # Smoothed label = most common in buffer
            if prediction_buffer:
                smoothed_prediction = Counter(prediction_buffer).most_common(1)[0][0]
                builder.update(smoothed_prediction, current_confidence)

        # ── Draw overlays ─────────────────────────────────────────────
        if show_landmarks and results.multi_hand_landmarks:
            frame = visualizer.draw_landmarks(frame, results, tracker.mp_hands)

        frame = visualizer.draw_ui(
            frame,
            prediction      = smoothed_prediction,
            confidence      = current_confidence,
            sentence        = builder.sentence,
            word_history    = builder.word_history,
            fps             = fps,
            is_recording    = is_recording,
            show_stats      = show_stats,
            hand_count      = len(landmarks_list),
            frame_count     = frame_count,
        )

        # ── Recording ─────────────────────────────────────────────────
        if is_recording:
            recorder.write(frame)

        cv2.imshow("Sign Language Detection", frame)

        # ── Key handling ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            if is_recording:
                recorder.stop()
                is_recording = False
                print("[REC] Recording saved.")
            else:
                recorder.start(frame.shape)
                is_recording = True
                print("[REC] Recording started.")
        elif key == ord('s'):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"screenshots/screenshot_{ts}.png"
            os.makedirs("screenshots", exist_ok=True)
            cv2.imwrite(path, frame)
            print(f"[SNAP] Saved: {path}")
        elif key == ord('l'):
            show_landmarks = not show_landmarks
        elif key == ord('i'):
            show_stats = not show_stats
        elif key == ord('c'):
            builder.clear()
            prediction_buffer.clear()
        elif key == ord(' '):
            builder.add_space()

    cap.release()
    if is_recording:
        recorder.stop()
    cv2.destroyAllWindows()
    print("\n👋 Exited. Goodbye!")


if __name__ == "__main__":
    main()

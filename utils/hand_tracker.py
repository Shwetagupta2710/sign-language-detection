"""
Hand Tracker — works with mediapipe 0.10.x (Tasks API)
Falls back to legacy mp.solutions.hands if available.
"""

import cv2
import numpy as np


class HandTracker:
    def __init__(self, config: dict):
        self._use_tasks = False
        self._detector  = None
        self.mp_hands   = None
        self.mp_drawing = None

        # ── Try new Tasks API (mediapipe >= 0.10) ────────────────────
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
            import urllib.request, os

            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")
            if not os.path.exists(model_path):
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                print("[INFO] Downloading hand_landmarker.task model (~11MB)...")
                urllib.request.urlretrieve(url, model_path)
                print("[INFO] Model downloaded.")

            base_opts = mp_tasks.BaseOptions(model_asset_path=model_path)
            opts = vision.HandLandmarkerOptions(
                base_options=base_opts,
                num_hands=2,
                min_hand_detection_confidence=config.get("min_detection_confidence", 0.7),
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=config.get("min_tracking_confidence", 0.5),
            )
            self._detector  = vision.HandLandmarker.create_from_options(opts)
            self._use_tasks = True
            print("[INFO] Using mediapipe Tasks API (0.10.x)")

        except Exception as e:
            # ── Fallback: legacy solutions API ───────────────────────
            try:
                import mediapipe as mp
                self.mp_hands   = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=config.get("min_detection_confidence", 0.7),
                    min_tracking_confidence=config.get("min_tracking_confidence", 0.5),
                )
                print("[INFO] Using mediapipe legacy solutions API")
            except Exception as e2:
                raise RuntimeError(f"Could not init mediapipe: {e2}") from e2

        # mp.solutions removed in 0.10.x - drawing is handled by Visualizer directly

    def process(self, frame: np.ndarray):
        if self._use_tasks:
            return self._process_tasks(frame)
        return self._process_legacy(frame)

    def _process_tasks(self, frame: np.ndarray):
        import mediapipe as mp
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._detector.detect(mp_image)

        landmarks_list = []
        fake_results   = _FakeResults()

        if result.hand_landmarks:
            for i, hand_lm_list in enumerate(result.hand_landmarks):
                lm_array = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_lm_list], dtype=np.float32
                )
                raw_pixels = (lm_array[:, :2] * [w, h]).astype(int)
                x_min, y_min = raw_pixels.min(axis=0)
                x_max, y_max = raw_pixels.max(axis=0)
                pad = 20
                bbox = (
                    max(x_min - pad, 0), max(y_min - pad, 0),
                    min(x_max + pad, w) - max(x_min - pad, 0),
                    min(y_max + pad, h) - max(y_min - pad, 0),
                )
                label = "Right"
                if result.handedness and i < len(result.handedness):
                    label = result.handedness[i][0].category_name
                landmarks_list.append({
                    "landmarks": lm_array, "handedness": label,
                    "bbox": bbox, "raw_pixels": raw_pixels,
                })
                fake_results.add_hand(hand_lm_list, label)

        return fake_results, landmarks_list

    def _process_legacy(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                lm_array = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32
                )
                raw_pixels = (lm_array[:, :2] * [w, h]).astype(int)
                x_min, y_min = raw_pixels.min(axis=0)
                x_max, y_max = raw_pixels.max(axis=0)
                pad = 20
                bbox = (
                    max(x_min - pad, 0), max(y_min - pad, 0),
                    min(x_max + pad, w) - max(x_min - pad, 0),
                    min(y_max + pad, h) - max(y_min - pad, 0),
                )
                landmarks_list.append({
                    "landmarks": lm_array,
                    "handedness": handedness.classification[0].label,
                    "bbox": bbox, "raw_pixels": raw_pixels,
                })

        return results, landmarks_list

    def extract_features(self, landmarks: np.ndarray) -> np.ndarray:
        relative = landmarks - landmarks[0]
        span     = np.linalg.norm(relative, axis=1).max() + 1e-6
        return (relative / span).flatten().astype(np.float32)


# ── Compatibility wrappers so Visualizer.draw_landmarks() works ───────────────

class _FakeResults:
    def __init__(self):
        self.multi_hand_landmarks = []
        self.multi_handedness     = []

    def add_hand(self, lm_list, label):
        self.multi_hand_landmarks.append(_FakeLandmarks(lm_list))
        self.multi_handedness.append(_FakeHandedness(label))

class _FakeLandmarks:
    def __init__(self, lm_list):
        self.landmark = lm_list

class _FakeHandedness:
    def __init__(self, label):
        self.classification = [type("C", (), {"label": label})()]

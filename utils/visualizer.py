"""
Visualizer — pure OpenCV, no mediapipe.solutions dependency
"""

import cv2
import numpy as np
import time

# MediaPipe 21-point hand connection pairs
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

PALETTE = {
    "bg_dark" : (18,  18,  18),
    "accent"  : (0,   220, 130),
    "accent2" : (0,   160, 255),
    "white"   : (255, 255, 255),
    "grey"    : (140, 140, 140),
    "red"     : (50,  50,  230),
    "lm_point": (0,   255, 200),
    "lm_line" : (255, 200,   0),
}

FONT      = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN


class Visualizer:
    def __init__(self):
        self._blink_state = True
        self._last_blink  = time.time()

    # ── Landmarks (pure OpenCV) ───────────────────────────────────────────────

    def draw_landmarks(self, frame, results, mp_hands=None):
        if not results.multi_hand_landmarks:
            return frame
        h, w = frame.shape[:2]
        for hand_lm in results.multi_hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm.landmark]
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], PALETTE["lm_line"], 2, cv2.LINE_AA)
            for cx, cy in pts:
                cv2.circle(frame, (cx, cy), 5, PALETTE["lm_point"], -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 0), 1, cv2.LINE_AA)
        return frame

    # ── Main UI ───────────────────────────────────────────────────────────────

    def draw_ui(self, frame, prediction, confidence, sentence, word_history,
                fps, is_recording, show_stats, hand_count, frame_count):
        h, w = frame.shape[:2]

        # Top bar
        self._overlay_rect(frame, 0, 0, w, 80, (10, 10, 10), alpha=0.65)
        cv2.putText(frame, "Sign Language Detection", (16, 50),
                    FONT, 1.0, PALETTE["accent"], 2, cv2.LINE_AA)

        if prediction:
            self._draw_prediction_box(frame, prediction, confidence, w, h)

        self._draw_sentence_panel(frame, sentence, word_history, h, w)

        if show_stats:
            self._draw_stats(frame, fps, hand_count, frame_count, w)

        if is_recording:
            self._draw_recording_indicator(frame)

        if hand_count == 0:
            msg = "No hand detected  —  show your hand to the camera"
            cv2.putText(frame, msg, (w // 2 - 290, h // 2),
                        FONT, 0.7, PALETTE["grey"], 1, cv2.LINE_AA)

        return frame

    def _draw_prediction_box(self, frame, label, confidence, w, h):
        bx, by, bw, bh = w // 2 - 130, 100, 260, 160
        self._overlay_rect(frame, bx, by, bw, bh, (20, 20, 20), alpha=0.75,
                           border_color=PALETTE["accent"], border_thick=2)
        font_scale = 3.5 if len(label) == 1 else 1.0
        tx = bx + bw // 2
        ty = by + 100 if len(label) == 1 else by + 85
        (tw, _), _ = cv2.getTextSize(label, FONT, font_scale, 3)
        cv2.putText(frame, label, (tx - tw // 2, ty),
                    FONT, font_scale, PALETTE["white"], 3, cv2.LINE_AA)
        bar_x, bar_y = bx + 15, by + bh - 28
        bar_w = bw - 30
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (50, 50, 50), -1)
        fill_w = int(bar_w * min(confidence, 1.0))
        colour = PALETTE["accent"] if confidence >= 0.85 else PALETTE["accent2"]
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 12), colour, -1)
        cv2.putText(frame, f"{confidence*100:.0f}%", (bar_x + bar_w + 6, bar_y + 11),
                    FONT_MONO, 1.1, PALETTE["grey"], 1, cv2.LINE_AA)

    def _draw_sentence_panel(self, frame, sentence, word_history, h, w):
        ph = 100
        self._overlay_rect(frame, 0, h - ph, w, ph, (10, 10, 10), alpha=0.70)
        history_str = "  >  ".join(list(word_history)[-8:])
        cv2.putText(frame, history_str, (16, h - ph + 24),
                    FONT_MONO, 1.0, PALETTE["grey"], 1, cv2.LINE_AA)
        display = sentence if len(sentence) <= 60 else "..." + sentence[-57:]
        cv2.putText(frame, display or "-- say something --", (16, h - 20),
                    FONT, 0.85, PALETTE["white"], 2, cv2.LINE_AA)

    def _draw_stats(self, frame, fps, hand_count, frame_count, w):
        lines = [f"FPS      {fps:.1f}", f"Hands    {hand_count}", f"Frames   {frame_count}"]
        bx = w - 190
        self._overlay_rect(frame, bx - 8, 6, 188, 88, (10, 10, 10), alpha=0.60)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (bx, 30 + i * 26),
                        FONT_MONO, 1.1, PALETTE["grey"], 1, cv2.LINE_AA)

    def _draw_recording_indicator(self, frame):
        now = time.time()
        if now - self._last_blink > 0.5:
            self._blink_state = not self._blink_state
            self._last_blink  = now
        if self._blink_state:
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, PALETTE["red"], -1)
            cv2.putText(frame, "REC", (frame.shape[1] - 75, 38),
                        FONT, 0.55, PALETTE["red"], 1, cv2.LINE_AA)

    @staticmethod
    def _overlay_rect(frame, x, y, w, h, color, alpha=0.6,
                      border_color=None, border_thick=0):
        sub = frame[y:y+h, x:x+w]
        if sub.size == 0:
            return
        rect = np.full_like(sub, color)
        cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
        if border_color and border_thick:
            cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, border_thick)

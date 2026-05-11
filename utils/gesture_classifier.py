"""
Gesture Classifier
==================
Rule-based + heuristic classifier for ASL hand signs.

Supports:
  • Full alphabet A–Z
  • Common words: HELLO, THANK YOU, YES, NO, I LOVE YOU, PLEASE, SORRY
  • Confidence scoring per gesture

Landmark indices (MediaPipe 21-point model):
  0  = WRIST
  1–4  = THUMB  (CMC→TIP)
  5–8  = INDEX  (MCP→TIP)
  9–12 = MIDDLE (MCP→TIP)
  13–16= RING   (MCP→TIP)
  17–20= PINKY  (MCP→TIP)
"""

import numpy as np


# ── Helper geometry functions ──────────────────────────────────────────────────

def _tip(lm, finger):
    """Tip landmark index: thumb=4, index=8, middle=12, ring=16, pinky=20."""
    return [4, 8, 12, 16, 20][finger]

def _pip(lm, finger):
    """PIP (second joint) index."""
    return [3, 6, 10, 14, 18][finger]

def _mcp(lm, finger):
    """MCP (knuckle) index."""
    return [2, 5, 9, 13, 17][finger]

def _dist(lm, a, b):
    return float(np.linalg.norm(lm[a, :2] - lm[b, :2]))

def _is_extended(lm, finger):
    """True if finger tip is above (lower y value) its MCP."""
    tip = _tip(lm, finger)
    pip = _pip(lm, finger)
    mcp = _mcp(lm, finger)
    # For thumb use x-axis comparison; for others use y-axis
    if finger == 0:
        return lm[tip, 0] < lm[mcp, 0] if lm[5, 0] > lm[17, 0] else lm[tip, 0] > lm[mcp, 0]
    return lm[tip, 1] < lm[pip, 1]

def _is_curled(lm, finger):
    return not _is_extended(lm, finger)

def _fingers_extended(lm):
    """Returns list of booleans [thumb, index, middle, ring, pinky]."""
    return [_is_extended(lm, i) for i in range(5)]

def _touching(lm, a, b, thresh=0.07):
    return _dist(lm, a, b) < thresh


# ── Main Classifier ────────────────────────────────────────────────────────────

class GestureClassifier:
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def classify(self, landmarks: np.ndarray):
        """
        landmarks: np.ndarray shape (21, 3), normalized 0-1
        Returns: (label: str | None, confidence: float)
        """
        # Normalize: wrist-relative + scale
        lm = landmarks.copy()
        lm -= lm[0]
        span = np.linalg.norm(lm, axis=1).max() + 1e-6
        lm /= span

        # Try each rule set; collect (label, confidence) pairs
        candidates = []

        # --- Common words first (higher priority) ---
        for fn in [
            self._hello, self._thank_you, self._i_love_you,
            self._yes, self._no, self._please, self._sorry,
        ]:
            label, conf = fn(lm)
            if label:
                candidates.append((label, conf))

        # --- Alphabet ---
        for fn in [
            self._a, self._b, self._c, self._d, self._e,
            self._f, self._g, self._h, self._i, self._j,
            self._k, self._l, self._m, self._n, self._o,
            self._p, self._q, self._r, self._s, self._t,
            self._u, self._v, self._w, self._x, self._y, self._z,
        ]:
            label, conf = fn(lm)
            if label:
                candidates.append((label, conf))

        if not candidates:
            return None, 0.0

        # Pick best
        best = max(candidates, key=lambda x: x[1])
        if best[1] >= self.threshold:
            return best
        return None, best[1]

    # ── Common word gestures ─────────────────────────────────────────────────

    def _hello(self, lm):
        """Open flat hand, all fingers extended, thumb out."""
        ext = _fingers_extended(lm)
        if all(ext):
            return "HELLO", 0.90
        return None, 0.0

    def _thank_you(self, lm):
        """Flat hand, fingers together, palm facing away (tips point up & forward)."""
        ext = _fingers_extended(lm)
        if ext[1] and ext[2] and ext[3] and ext[4] and not ext[0]:
            # Middle finger roughly at wrist height or above
            if lm[12, 1] < -0.3:
                return "THANK YOU", 0.85
        return None, 0.0

    def _i_love_you(self, lm):
        """ILY: thumb + index + pinky extended."""
        ext = _fingers_extended(lm)
        if ext[0] and ext[1] and not ext[2] and not ext[3] and ext[4]:
            return "I LOVE YOU", 0.92

        return None, 0.0

    def _yes(self, lm):
        """Fist nodding — closed fist (A-shape)."""
        ext = _fingers_extended(lm)
        if not any(ext[1:]):
            return "YES", 0.80
        return None, 0.0

    def _no(self, lm):
        """Index + middle extended, snapping together."""
        ext = _fingers_extended(lm)
        if ext[1] and ext[2] and not ext[3] and not ext[4]:
            close = _touching(lm, 8, 12, 0.10)
            if close:
                return "NO", 0.82
        return None, 0.0

    def _please(self, lm):
        """Flat hand on chest (middle finger extended, others curled slightly)."""
        ext = _fingers_extended(lm)
        if ext[1] and ext[2] and ext[3] and ext[4] and ext[0]:
            if lm[9, 1] > -0.1:  # hand lower (near chest level)
                return "PLEASE", 0.78
        return None, 0.0

    def _sorry(self, lm):
        """Fist with thumb extended, circular motion — static: fist + thumb up."""
        ext = _fingers_extended(lm)
        if ext[0] and not ext[1] and not ext[2] and not ext[3] and not ext[4]:
            if lm[4, 1] < -0.2:
                return "SORRY", 0.80
        return None, 0.0

    # ── ASL Alphabet ─────────────────────────────────────────────────────────

    def _a(self, lm):
        ext = _fingers_extended(lm)
        # Closed fist, thumb rests on side
        if not ext[1] and not ext[2] and not ext[3] and not ext[4]:
            if lm[4, 0] > lm[3, 0]:   # thumb slightly to side
                return "A", 0.82
        return None, 0.0

    def _b(self, lm):
        ext = _fingers_extended(lm)
        if ext[1] and ext[2] and ext[3] and ext[4] and not ext[0]:
            return "B", 0.85
        return None, 0.0

    def _c(self, lm):
        # Curved fingers forming C shape
        ext = _fingers_extended(lm)
        # Partially curled: tips lower than MCPs but not fully curled
        if not ext[1] and not ext[2] and not ext[3] and not ext[4]:
            # Check that fingers are not fully curled (C = half curl)
            if lm[8, 1] > -0.2 and lm[8, 1] < 0.1:
                return "C", 0.78
        return None, 0.0

    def _d(self, lm):
        ext = _fingers_extended(lm)
        # Index up, others curled, thumb touches middle
        if ext[1] and not ext[2] and not ext[3] and not ext[4]:
            if _touching(lm, 4, 12, 0.12):
                return "D", 0.80
        return None, 0.0

    def _e(self, lm):
        # All fingers curled, thumb tucked under
        ext = _fingers_extended(lm)
        if not any(ext):
            if lm[4, 1] > lm[8, 1]:   # thumb below index tip
                return "E", 0.78
        return None, 0.0

    def _f(self, lm):
        ext = _fingers_extended(lm)
        # OK-ish: index+thumb touching, others extended
        if not ext[1] and ext[2] and ext[3] and ext[4]:
            if _touching(lm, 4, 8, 0.10):
                return "F", 0.82
        return None, 0.0

    def _g(self, lm):
        ext = _fingers_extended(lm)
        # Index pointing sideways, thumb parallel
        if ext[1] and not ext[2] and not ext[3] and not ext[4]:
            if abs(lm[8, 0] - lm[5, 0]) > 0.15:  # horizontal extension
                return "G", 0.78
        return None, 0.0

    def _h(self, lm):
        ext = _fingers_extended(lm)
        if ext[1] and ext[2] and not ext[3] and not ext[4]:
            # Both roughly horizontal
            return "H", 0.78
        return None, 0.0

    def _i(self, lm):
        ext = _fingers_extended(lm)
        if not ext[0] and not ext[1] and not ext[2] and not ext[3] and ext[4]:
            return "I", 0.85
        return None, 0.0

    def _j(self, lm):
        # J is I + motion; static approximation = pinky extended + slight tilt
        ext = _fingers_extended(lm)
        if ext[4] and not ext[1] and not ext[2] and not ext[3]:
            if lm[20, 0] < lm[17, 0] - 0.1:   # pinky tilted left
                return "J", 0.75
        return None, 0.0

    def _k(self, lm):
        ext = _fingers_extended(lm)
        if ext[0] and ext[1] and ext[2] and not ext[3] and not ext[4]:
            return "K", 0.80
        return None, 0.0

    def _l(self, lm):
        ext = _fingers_extended(lm)
        if ext[0] and ext[1] and not ext[2] and not ext[3] and not ext[4]:
            # L shape: index up, thumb horizontal
            if lm[8, 1] < -0.3 and lm[4, 0] > 0.1:
                return "L", 0.88
        return None, 0.0

    def _m(self, lm):
        ext = _fingers_extended(lm)
        # Three fingers curled over thumb
        if not any(ext):
            if lm[4, 1] > lm[12, 1]:  # thumb tucked under fingers
                return "M", 0.75
        return None, 0.0

    def _n(self, lm):
        ext = _fingers_extended(lm)
        if not any(ext):
            if lm[4, 1] > lm[8, 1]:   # slight variant of M
                return "N", 0.74
        return None, 0.0

    def _o(self, lm):
        ext = _fingers_extended(lm)
        # All fingers curved to form O with thumb
        if not any(ext):
            if _touching(lm, 4, 8, 0.12):
                return "O", 0.82
        return None, 0.0

    def _p(self, lm):
        ext = _fingers_extended(lm)
        if ext[0] and ext[1] and ext[2] and not ext[3] and not ext[4]:
            if lm[8, 1] > 0:   # index pointing down
                return "P", 0.78
        return None, 0.0

    def _q(self, lm):
        ext = _fingers_extended(lm)
        if ext[0] and ext[1] and not ext[2] and not ext[3] and not ext[4]:
            if lm[8, 1] > 0:   # index + thumb pointing down
                return "Q", 0.76
        return None, 0.0

    def _r(self, lm):
        ext = _fingers_extended(lm)
        if ext[1] and ext[2] and not ext[3] and not ext[4]:
            # Crossed fingers
            if abs(lm[8, 0] - lm[12, 0]) < 0.05:
                return "R", 0.80
        return None, 0.0

    def _s(self, lm):
        ext = _fingers_extended(lm)
        # Closed fist, thumb over fingers
        if not any(ext):
            if lm[4, 1] < lm[8, 1]:  # thumb above fingers
                return "S", 0.80
        return None, 0.0

    def _t(self, lm):
        ext = _fingers_extended(lm)
        if not any(ext[1:]):
            if lm[4, 0] < lm[8, 0]:   # thumb between index and middle
                return "T", 0.76
        return None, 0.0

    def _u(self, lm):
        ext = _fingers_extended(lm)
        if not ext[0] and ext[1] and ext[2] and not ext[3] and not ext[4]:
            if abs(lm[8, 0] - lm[12, 0]) < 0.05:  # fingers together
                return "U", 0.82
        return None, 0.0

    def _v(self, lm):
        ext = _fingers_extended(lm)
        if not ext[0] and ext[1] and ext[2] and not ext[3] and not ext[4]:
            if abs(lm[8, 0] - lm[12, 0]) > 0.06:  # fingers spread (V)
                return "V", 0.85
        return None, 0.0

    def _w(self, lm):
        ext = _fingers_extended(lm)
        if not ext[0] and ext[1] and ext[2] and ext[3] and not ext[4]:
            return "W", 0.83
        return None, 0.0

    def _x(self, lm):
        ext = _fingers_extended(lm)
        # Index hooked
        if not ext[0] and not ext[2] and not ext[3] and not ext[4]:
            if lm[8, 1] > lm[6, 1] and lm[6, 1] < lm[5, 1]:
                return "X", 0.76
        return None, 0.0

    def _y(self, lm):
        ext = _fingers_extended(lm)
        if ext[0] and not ext[1] and not ext[2] and not ext[3] and ext[4]:
            return "Y", 0.88
        return None, 0.0

    def _z(self, lm):
        # Z is drawn in air; static: index extended, pointing sideways
        ext = _fingers_extended(lm)
        if ext[1] and not ext[2] and not ext[3] and not ext[4]:
            if lm[8, 0] > 0.2:   # index pointing strongly to the side
                return "Z", 0.74
        return None, 0.0

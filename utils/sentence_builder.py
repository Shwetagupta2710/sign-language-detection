"""
Sentence Builder
================
Assembles detected gestures into words and full sentences.
Uses a pause-based word boundary: if no gesture is detected for
`pause_threshold` seconds, the current letter sequence → new word.
"""

import time
from collections import deque


class SentenceBuilder:
    def __init__(self, pause_threshold: float = 2.0, min_confidence: float = 0.75):
        self.pause_threshold = pause_threshold
        self.min_confidence  = min_confidence

        self.sentence     : str         = ""
        self.current_word : str         = ""
        self.word_history : deque       = deque(maxlen=20)

        self._last_sign_time  : float = time.time()
        self._last_label      : str   = ""
        self._label_hold_time : float = 0.0
        self._letter_added    : bool  = False
        self._hold_threshold  : float = 0.8   # seconds to hold before adding

    def update(self, label: str | None, confidence: float):
        now = time.time()

        if label is None or confidence < self.min_confidence:
            # Check for pause → word boundary
            elapsed = now - self._last_sign_time
            if elapsed > self.pause_threshold and self.current_word:
                self._commit_word()
            self._last_label      = ""
            self._label_hold_time = now
            self._letter_added    = False
            return

        # New label seen
        if label != self._last_label:
            self._last_label      = label
            self._label_hold_time = now
            self._letter_added    = False

        # Add letter if held long enough and not already added
        held = now - self._label_hold_time
        if held >= self._hold_threshold and not self._letter_added:
            if label.isalpha() and len(label) == 1:
                self.current_word += label
            else:
                # It's a full word gesture (HELLO, ILY, etc.)
                self._commit_word()
                self.word_history.append(label)
                self.sentence += ("" if not self.sentence else " ") + label
            self._letter_added    = True
            self._last_sign_time  = now

        self._last_sign_time = now

    def _commit_word(self):
        if self.current_word.strip():
            word = self.current_word.strip()
            self.word_history.append(word)
            self.sentence += ("" if not self.sentence else " ") + word
        self.current_word = ""

    def add_space(self):
        self._commit_word()

    def backspace(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence:
            parts = self.sentence.rsplit(" ", 1)
            self.sentence = parts[0]
            if len(parts) > 1:
                self.current_word = parts[1]

    def clear(self):
        self.sentence     = ""
        self.current_word = ""
        self.word_history.clear()

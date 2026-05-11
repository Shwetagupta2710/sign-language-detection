# 🤟 Advanced Sign Language Detection System

Real-time ASL detection using **MediaPipe** + **OpenCV**, with sentence building,
confidence scoring, recording, and more — all in pure Python.

---

## 📁 Project Structure

```
sign_language_detection/
│
├── main.py                    ← Entry point — run this
│
├── utils/
│   ├── hand_tracker.py        ← MediaPipe hand landmark extraction
│   ├── gesture_classifier.py  ← Rule-based ASL classifier (A–Z + words)
│   ├── sentence_builder.py    ← Assembles letters → words → sentences
│   ├── visualizer.py          ← All on-screen UI overlays
│   └── recorder.py            ← Video recording to .mp4
│
├── requirements.txt
├── .vscode/launch.json        ← VS Code run config
│
├── screenshots/               ← Auto-created on first screenshot
└── recordings/                ← Auto-created on first recording
```

---

## ⚙️ Setup (Step-by-Step)

### 1. Install Python 3.10 or 3.11

Download from https://www.python.org/downloads/
✅ Check **"Add Python to PATH"** during installation.

### 2. Install VS Code

Download from https://code.visualstudio.com/
Then install the **Python extension** (Ctrl+Shift+X → search "Python" → Install).

### 3. Open the project in VS Code

```
File → Open Folder → select sign_language_detection/
```

### 4. Create a virtual environment

Open the integrated terminal (Ctrl+\`):

```bash
python -m venv venv
```

Activate it:
- **Windows:**  `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

You'll see `(venv)` in the terminal prompt.

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
| Package | What it does |
|---------|-------------|
| `opencv-python` | Camera capture + drawing |
| `mediapipe` | 21-point hand landmark detection |
| `numpy` | Fast array maths |

### 6. Run the app

```bash
python main.py
```

Or press **F5** in VS Code (uses `.vscode/launch.json`).

---

## 🎮 Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Start / Stop video recording |
| `S` | Screenshot |
| `L` | Toggle hand landmark overlay |
| `I` | Toggle stats panel (FPS, frame count) |
| `C` | Clear sentence |
| `SPACE` | Force word boundary (add space) |

---

## ✋ Supported Signs

### Full ASL Alphabet
`A B C D E F G H I J K L M N O P Q R S T U V W X Y Z`

### Common Words / Phrases
| Gesture | Description |
|---------|-------------|
| `HELLO` | Open flat hand, all 5 fingers extended |
| `THANK YOU` | Flat hand, palm away, fingers up |
| `I LOVE YOU` | Thumb + index + pinky extended (ILY) |
| `YES` | Closed fist |
| `NO` | Index + middle touching/snapping |
| `PLEASE` | Flat hand, lower position |
| `SORRY` | Closed fist + thumb up |

---

## 🔬 How It Works

```
Camera Frame
     │
     ▼
MediaPipe Hands  ──► 21 landmarks (x, y, z) per hand
     │
     ▼
Gesture Classifier  ──► Rule-based finger geometry checks
     │                   → label + confidence score
     ▼
Prediction Buffer  ──► 10-frame rolling majority vote (smoothing)
     │
     ▼
Sentence Builder  ──► Hold 0.8s to add letter / 2s pause = new word
     │
     ▼
Visualizer  ──► Overlays on frame → cv2.imshow()
```

---

## 🚀 Advanced Features

| Feature | Location |
|---------|----------|
| Two-hand detection | `hand_tracker.py` — `max_num_hands=2` |
| Smoothed predictions | `main.py` — 10-frame rolling buffer |
| Confidence scoring | `gesture_classifier.py` — per-gesture float 0–1 |
| Hold-to-confirm | `sentence_builder.py` — 0.8 s hold threshold |
| Pause → word boundary | `sentence_builder.py` — 2.0 s silence |
| MP4 recording | `recorder.py` |
| Screenshot | `main.py` key handler |
| Semi-transparent UI | `visualizer.py` — alpha blending |

---

## 🛠️ Customization Tips

**Adjust sensitivity** — edit `main.py` CONFIG dict:
```python
"confidence_threshold": 0.75,   # lower = more sensitive
"prediction_smoothing": 10,     # higher = smoother but slower
"sentence_pause_sec":   2.0,    # pause before new word
```

**Add new gestures** — add a method to `GestureClassifier`:
```python
def _thumbs_up(self, lm):
    ext = _fingers_extended(lm)
    if ext[0] and not any(ext[1:]):
        return "THUMBS UP", 0.90
    return None, 0.0
```
Then call it inside `classify()`.

**Change camera** — edit `camera_index` in CONFIG (0 = default, 1 = external).

---

## ❓ Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with venv active |
| Camera not opening | Change `camera_index` to `1` or `2` in CONFIG |
| Low FPS | Lower `width`/`height` in CONFIG or close other apps |
| mediapipe install fails | Use Python 3.10 or 3.11 (not 3.12+) |
| Signs not detected | Ensure good lighting; keep hand 30–60 cm from camera |

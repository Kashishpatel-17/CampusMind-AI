"""
Run this inside your venv to see exactly what is failing:
  .venv\Scripts\activate
  python diagnose.py
"""
import sys
print(f"Python: {sys.version}\n")

tests = [
    ("numpy",       "import numpy as np; print('numpy', np.__version__)"),
    ("librosa",     "import librosa; print('librosa', librosa.__version__)"),
    ("soundfile",   "import soundfile; print('soundfile OK')"),
    ("cv2",         "import cv2; print('opencv', cv2.__version__)"),
    ("mediapipe",   "import mediapipe as mp; print('mediapipe', mp.__version__)"),
    ("whisper",     "import whisper; print('whisper OK')"),
    ("torch",       "import torch; print('torch', torch.__version__)"),
    ("fastapi",     "import fastapi; print('fastapi', fastapi.__version__)"),
    ("httpx",       "import httpx; print('httpx', httpx.__version__)"),
    ("anthropic",   "import anthropic; print('anthropic OK')"),
    ("dotenv",      "from dotenv import load_dotenv; print('dotenv OK')"),
    ("multipart",   "import multipart; print('multipart OK')"),
]

print("=" * 50)
all_ok = True
for name, code in tests:
    try:
        exec(code)
    except Exception as e:
        print(f"❌ {name} FAILED: {e}")
        all_ok = False

print("=" * 50)
if all_ok:
    print("✅ All packages OK — the crash is elsewhere")
else:
    print("⬆ Fix the FAILED packages above first")
    print("\nRun this to reinstall everything cleanly:")
    print("  pip install --upgrade pip")
    print("  pip install -r requirements.txt --force-reinstall")

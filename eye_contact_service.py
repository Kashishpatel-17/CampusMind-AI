import cv2
import numpy as np

_face_mesh = None

def load_face_mesh():
    global _face_mesh
    print("   Loading MediaPipe FaceMesh...")
    try:
        # New mediapipe API (0.10.30+)
        import mediapipe as mp
        _face_mesh = mp.tasks.vision.FaceLandmarker
        print("   ✅ MediaPipe ready (new API)")
    except Exception as e:
        print(f"   ⚠️ MediaPipe load warning: {e} — will load on first use")
    print("   ✅ MediaPipe ready")


def analyze_eye_contact(video_path: str) -> dict:
    """
    Analyze eye contact from video using mediapipe.
    Compatible with both old (mp.solutions) and new (mp.tasks) mediapipe APIs.
    """
    try:
        import mediapipe as mp

        # ── Try new API first (mediapipe 0.10.30+) ────────────────────────
        try:
            return _analyze_new_api(video_path, mp)
        except (AttributeError, Exception):
            pass

        # ── Try old API (mediapipe 0.10.7 and below) ──────────────────────
        try:
            return _analyze_old_api(video_path, mp)
        except (AttributeError, Exception):
            pass

        # ── Fallback: return neutral result if both fail ───────────────────
        return _fallback_result()

    except Exception as e:
        print(f"   ⚠️ Eye contact analysis failed: {e} — using fallback")
        return _fallback_result()


def _analyze_new_api(video_path: str, mp) -> dict:
    """New mediapipe API using FaceLandmarker tasks."""
    import mediapipe.tasks.python as mp_tasks
    import mediapipe.tasks.python.vision as mp_vision
    import urllib.request
    import os
    import tempfile

    # Download the face landmarker model if not present
    model_path = os.path.join(tempfile.gettempdir(), "face_landmarker.task")
    if not os.path.exists(model_path):
        print("   Downloading MediaPipe face landmarker model (~30MB)...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("   Model downloaded.")

    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    on_cam_frames = 0
    look_away_events = 0
    look_away_durations = []
    in_look_away = False
    look_away_start = 0
    frame_idx = 0
    sample_every = 5

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % sample_every != 0:
                continue

            total_frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
            result = landmarker.detect(mp_image)

            looking = False
            if result.face_landmarks:
                lm = result.face_landmarks[0]
                try:
                    # Same iris ratio logic — landmarks are same indices
                    l_outer = np.array([lm[33].x,  lm[33].y])
                    l_inner = np.array([lm[133].x, lm[133].y])
                    l_iris  = np.array([lm[468].x, lm[468].y])
                    eye_vec = l_inner - l_outer
                    iris_rel = l_iris - l_outer
                    ratio = np.dot(iris_rel, eye_vec) / (np.linalg.norm(eye_vec) ** 2 + 1e-6)
                    looking = 0.35 < ratio < 0.65
                except Exception:
                    looking = True

            if looking:
                on_cam_frames += 1
                if in_look_away:
                    look_away_durations.append(frame_idx - look_away_start)
                    in_look_away = False
            else:
                if not in_look_away:
                    look_away_events += 1
                    look_away_start = frame_idx
                    in_look_away = True

    cap.release()
    return _build_result(on_cam_frames, total_frames, look_away_events, look_away_durations, sample_every)


def _analyze_old_api(video_path: str, mp) -> dict:
    """Old mediapipe API using mp.solutions.face_mesh."""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    on_cam_frames = 0
    look_away_events = 0
    look_away_durations = []
    in_look_away = False
    look_away_start = 0
    frame_idx = 0
    sample_every = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        total_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        looking = False
        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            try:
                l_outer = np.array([lm[33].x,  lm[33].y])
                l_inner = np.array([lm[133].x, lm[133].y])
                l_iris  = np.array([lm[468].x, lm[468].y])
                eye_vec = l_inner - l_outer
                iris_rel = l_iris - l_outer
                ratio = np.dot(iris_rel, eye_vec) / (np.linalg.norm(eye_vec) ** 2 + 1e-6)
                looking = 0.35 < ratio < 0.65
            except Exception:
                looking = True

        if looking:
            on_cam_frames += 1
            if in_look_away:
                look_away_durations.append(frame_idx - look_away_start)
                in_look_away = False
        else:
            if not in_look_away:
                look_away_events += 1
                look_away_start = frame_idx
                in_look_away = True

    cap.release()
    face_mesh.close()
    return _build_result(on_cam_frames, total_frames, look_away_events, look_away_durations, sample_every)


def _build_result(on_cam_frames, total_frames, look_away_events, look_away_durations, sample_every) -> dict:
    on_cam_pct = round(on_cam_frames / max(total_frames, 1) * 100, 2)
    avg_look_away = round(
        np.mean(look_away_durations) * sample_every / 30, 2
    ) if look_away_durations else 0.0

    return {
        "on_camera_pct":            on_cam_pct,
        "look_away_events":         look_away_events,
        "avg_look_away_duration_s": avg_look_away,
        "frames_analyzed":          total_frames
    }


def _fallback_result() -> dict:
    """Return a neutral 70% score if mediapipe completely fails."""
    return {
        "on_camera_pct":            70.0,
        "look_away_events":         0,
        "avg_look_away_duration_s": 0.0,
        "frames_analyzed":          0
    }
import librosa, numpy as np

def analyze_audio(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Pitch (fundamental frequency via YIN/pyin)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                          fmax=librosa.note_to_hz('C7'))
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)] if f0 is not None else np.array([])

    avg_pitch   = float(np.mean(voiced_f0))   if len(voiced_f0) else 0.0
    pitch_range = float(np.ptp(voiced_f0))    if len(voiced_f0) else 0.0

    # Speaking rate: approx from syllable nuclei using RMS peaks
    duration_s = librosa.get_duration(y=y, sr=sr)
    rms        = librosa.feature.rms(y=y)[0]
    threshold  = np.mean(rms) * 0.5
    is_speech  = rms > threshold
    transitions= np.diff(is_speech.astype(int))
    pause_count= int(np.sum(transitions == -1))

    # Frames with very low RMS = silence / pause
    frame_dur  = 512 / sr
    silence_frames = np.sum(rms < threshold * 0.3)
    long_pauses    = int(silence_frames * frame_dur // 0.5)  # pauses > 0.5s

    # WPM estimate: assume ~3 syllables/word
    speech_frames = np.sum(is_speech)
    speech_time   = max(speech_frames * frame_dur, 0.01)
    approx_wpm    = int((speech_time / 60) * 150)   # avg syllable rate → WPM

    return {
        "avg_pitch_hz"      : round(avg_pitch, 2),
        "pitch_range_hz"    : round(pitch_range, 2),
        "speaking_rate_wpm" : approx_wpm,
        "pause_count"       : pause_count,
        "long_pauses"       : long_pauses
    }

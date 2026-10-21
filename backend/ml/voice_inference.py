"""MindSetu Voice ML Inference Engine.

Performs acoustic feature extraction, paralinguistic classification,
audio quality verification, and feature attribution.
"""

from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .voice_feature_extractor import decode_wav_bytes, extract_acoustic_features, ACOUSTIC_FEATURE_NAMES


MODEL_PATH = Path(__file__).resolve().parent / "voice_depression_classifier.joblib"


@lru_cache(maxsize=1)
def load_voice_model():
    """Loads the serialized voice ML bundle."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Voice model artifact not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def analyze_voice_recording(wav_bytes: bytes, session_id: str | None = None) -> dict:
    """Processes raw WAV audio bytes, extracts acoustic features, and executes model inference."""
    # 1. Decode WAV
    try:
        audio, sample_rate = decode_wav_bytes(wav_bytes)
    except Exception as e:
        raise ValueError(f"Failed to decode audio file. Please ensure it is a valid WAV audio recording: {str(e)}")

    # 2. Extract Acoustic & Prosodic Features
    features, diagnostics = extract_acoustic_features(audio, sample_rate)

    # 3. Load Model Bundle
    bundle = load_voice_model()
    pipeline = bundle["pipeline"]
    feature_names = bundle["features"]

    # 4. Predict — raw model probability scaled to Acoustic Strain Index 0–100
    df = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names)
    raw_prob = float(pipeline.predict_proba(df)[0, 1])
    acoustic_strain_signal = round(raw_prob * 100.0, 1)

    # Keep legacy keys for backwards compatibility with frontend and tests
    depression_signal = acoustic_strain_signal

    # 5. Vocal Energy & Dynamic Range Index — multi-feature composite
    # Combines pitch variability, speech cadence, and spectral energy for a separate dynamic signal
    pitch_std = features.get("pitch_f0_std", 15.0)
    pitch_rng = features.get("pitch_range", 40.0)
    speech_rate = features.get("speech_rate_estimate", 3.0)
    rms = features.get("rms_energy", 0.06)
    spec_flux = features.get("spectral_flux", 0.05)

    # Higher values → more dynamic/resilient; lower → strained/fatigued
    norm_pitch_std = min(1.0, max(0.0, (pitch_std - 5.0) / 35.0))     # 5–40 Hz → 0–1
    norm_pitch_rng = min(1.0, max(0.0, (pitch_rng - 10.0) / 65.0))    # 10–75 Hz → 0–1
    norm_rate      = min(1.0, max(0.0, (speech_rate - 1.5) / 4.0))    # 1.5–5.5 → 0–1
    norm_rms       = min(1.0, max(0.0, (rms - 0.02) / 0.12))          # 0.02–0.14 → 0–1
    norm_flux      = min(1.0, max(0.0, (spec_flux - 0.02) / 0.07))    # 0.02–0.09 → 0–1
    vocal_energy_composite = (norm_pitch_std * 0.30 + norm_pitch_rng * 0.25 +
                               norm_rate * 0.20 + norm_rms * 0.15 + norm_flux * 0.10)
    vocal_energy_index = round(vocal_energy_composite * 100.0, 1)

    # Keep legacy key for backwards compatibility
    trauma_signal = round((1.0 - vocal_energy_composite) * 100.0, 1)

    # 6. Top Acoustic Contributors / Explainability
    clf = pipeline.named_steps["classifier"]
    scaler = pipeline.named_steps["scaler"]
    importances = clf.feature_importances_

    x_scaled = scaler.transform(df)[0]
    contributors = []

    friendly_labels = {
        "pitch_range": "Dynamic Pitch Range",
        "pitch_f0_std": "Pitch Variability (Monotone vs Animated)",
        "pause_ratio": "Hesitation & Pause Proportion",
        "speech_rate_estimate": "Speech Cadence & Rate",
        "spectral_flux": "Spectral Modulation",
        "rms_energy": "Acoustic Vocal Energy",
        "spectral_centroid": "Vocal Tract Brightness",
        "energy_entropy": "Energy Distribution",
        "zero_crossing_rate": "Voicing Articulation",
    }

    for idx, feat in enumerate(feature_names):
        weight = float(importances[idx])
        dev = float(x_scaled[idx])
        impact = weight * abs(dev)
        direction = "increases signal" if dev > 0.3 else ("reduces signal" if dev < -0.3 else "neutral")
        label = friendly_labels.get(feat, feat.replace("_", " ").title())
        contributors.append({
            "feature": feat,
            "label": label,
            "measured_value": features[feat],
            "importance_weight": round(weight, 4),
            "impact_magnitude": round(impact, 4),
            "direction": direction,
        })

    contributors = sorted(contributors, key=lambda c: c["impact_magnitude"], reverse=True)[:5]

    # 7. Confidence Score based on audio quality & duration
    duration = diagnostics["duration_seconds"]
    quality = diagnostics["quality"]
    margin = abs(raw_prob - 0.5) * 2.0  # 0.0 to 1.0

    if quality == "good" and duration >= 5.0:
        confidence = round(min(0.92, 0.65 + margin * 0.27), 2)
    elif quality == "moderate":
        confidence = round(min(0.78, 0.50 + margin * 0.28), 2)
    else:
        confidence = round(min(0.58, 0.38 + margin * 0.20), 2)

    # 8. Welfare-Appropriate Signal Interpretation (non-clinical)
    if acoustic_strain_signal >= 70.0:
        strain_band = "elevated"
        interpretation = (
            f"Acoustic analysis detected constrained pitch modulation ({features.get('pitch_f0_std', 0):.1f} Hz std), "
            f"elevated pause proportion ({features.get('pause_ratio', 0)*100:.0f}%), "
            f"and reduced speech cadence ({features.get('speech_rate_estimate', 0):.1f} peaks/s). "
            "These vocal signals suggest the speaker may benefit from rest or a welfare check-in."
        )
    elif acoustic_strain_signal >= 45.0:
        strain_band = "moderate"
        interpretation = (
            f"Speech shows moderate pitch variation ({features.get('pitch_f0_std', 0):.1f} Hz std) "
            f"with a pause ratio of {features.get('pause_ratio', 0)*100:.0f}%. "
            "Some acoustic strain indicators are present — consider monitoring workload and rest patterns."
        )
    else:
        strain_band = "low"
        interpretation = (
            f"Speech reflects active pitch modulation ({features.get('pitch_f0_std', 0):.1f} Hz std) "
            f"and a healthy speech cadence ({features.get('speech_rate_estimate', 0):.1f} peaks/s). "
            "Vocal acoustic patterns indicate baseline engagement and vocal energy."
        )

    return {
        "status": "success",
        "session_id": session_id,
        # Primary welfare signal (non-clinical)
        "acoustic_strain_signal": acoustic_strain_signal,
        "strain_band": strain_band,
        "vocal_energy_index": vocal_energy_index,
        # Legacy keys kept for frontend backward compatibility
        "depression_signal": acoustic_strain_signal,
        "trauma_signal": trauma_signal,
        "trauma_status": "vocal_energy_composite",
        "confidence": confidence,
        "audio_quality": quality,
        "diagnostics": diagnostics,
        "features": features,
        "top_acoustic_contributors": contributors,
        "signal_interpretation": interpretation,
        "model_version": bundle.get("model_version", "1.0.0-voice-gbdt"),
        "clinical_diagnosis": False,
        "notice": "This voice analysis provides acoustic welfare signals only. It does not constitute a clinical diagnosis or a personnel decision."
    }


def generate_synthetic_demo_wav(scenario: str = "strained") -> bytes:
    """Generates a synthetic 16kHz PCM WAV byte stream for demonstration purposes."""
    import io
    import wave
    
    sample_rate = 16000
    duration = 4.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    if scenario == "strained":
        # Flatter pitch ~125 Hz with subtle harmonics, lower volume, slight pauses
        f0 = 125.0
        audio = 0.04 * np.sin(2 * np.pi * f0 * t) + 0.02 * np.sin(2 * np.pi * 2 * f0 * t)
        # Add pauses
        pause_mask = (t % 1.2) > 0.7
        audio[pause_mask] = 0.001 * np.random.randn(np.sum(pause_mask))
    elif scenario == "resilient":
        # Animated pitch modulated between 140 Hz and 190 Hz, energetic
        f0 = 155.0 + 35.0 * np.sin(2 * np.pi * 1.5 * t)
        phase = np.cumsum(2 * np.pi * f0 / sample_rate)
        audio = 0.08 * np.sin(phase) + 0.03 * np.sin(2 * phase)
        pause_mask = (t % 1.5) > 1.2
        audio[pause_mask] = 0.001 * np.random.randn(np.sum(pause_mask))
    else:
        # Moderate
        f0 = 140.0 + 15.0 * np.sin(2 * np.pi * 0.8 * t)
        phase = np.cumsum(2 * np.pi * f0 / sample_rate)
        audio = 0.06 * np.sin(phase)
        
    audio_int16 = (audio * 32767.0).astype(np.int16)
    
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
        
    return bio.getvalue()

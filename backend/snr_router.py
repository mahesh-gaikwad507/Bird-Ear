
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class SNRResult:
    snr_db: float
    route: str
    reason: str


def estimate_snr(audio_path: str) -> float:
    """
    Estimate SNR in dB using a simple frame-energy heuristic.

    This is a rough proxy, not a true measurement of signal and
    noise. It assumes quieter frames represent background noise.
    """
    audio, sample_rate = sf.read(audio_path, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    frame_length = int(sample_rate * 0.02)  # 20 ms
    if frame_length < 1:
        raise ValueError("Invalid audio sample rate.")

    frame_count = len(audio) // frame_length
    if frame_count < 2:
        raise ValueError("Audio is too short for SNR estimation.")

    audio = audio[: frame_count * frame_length]
    frames = audio.reshape(frame_count, frame_length)

    frame_power = np.mean(frames ** 2, axis=1) + 1e-12

    # Approximate the noise floor from the quietest 20% of frames.
    noise_power = np.percentile(frame_power, 20)
    signal_power = np.percentile(frame_power, 90)

    snr_db = 10.0 * np.log10(
        max(signal_power - noise_power, 1e-12)
        / max(noise_power, 1e-12)
    )

    return float(snr_db)


def route_audio(audio_path: str) -> SNRResult:
    """
    Route by estimated audio quality.

    These initial thresholds are configurable heuristics, not
    validated performance thresholds.
    """
    path = Path(audio_path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    snr_db = estimate_snr(str(path))

    if snr_db < 5:
        return SNRResult(
            snr_db, "both", "Low estimated SNR; request complementary analysis."
        )
    elif snr_db < 15:
        return SNRResult(
            snr_db, "vggish", "Moderate estimated SNR; extract audio features."
        )
    else:
        return SNRResult(
            snr_db, "birdnet", "High estimated SNR; prioritize BirdNET."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SNR-based audio router")
    parser.add_argument("audio_path", help="Path to a WAV audio recording")
    args = parser.parse_args()

    result = route_audio(args.audio_path)
    print(f"Estimated SNR: {result.snr_db:.2f} dB")
    print(f"Route: {result.route}")
    print(f"Reason: {result.reason}")

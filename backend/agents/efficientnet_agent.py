from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0


class EfficientNetAudioAgent:
    """Extract general-purpose features from audio spectrograms."""

    def __init__(self):
        self.model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=(224, 224, 3),
        )

        self.model.trainable = False

    def analyze(self, audio_path: str) -> np.ndarray:
        audio_path = Path(audio_path)

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        # Convert audio to mono and resample to 22,050 Hz.
        audio, _ = librosa.load(
            str(audio_path),
            sr=22050,
            mono=True,
        )

        if audio.size == 0:
            raise ValueError("Audio file contains no audio samples.")

        # Generate a log-mel spectrogram.
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            power=2.0,
        )

        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Normalize the spectrogram to an image-like range.
        image = (log_mel + 80.0) / 80.0
        image = np.clip(image, 0.0, 1.0)

        # Resize to the model's required spatial dimensions.
        image_tensor = tf.convert_to_tensor(
            image[np.newaxis, ..., np.newaxis],
            dtype=tf.float32,
        )

        image_tensor = tf.image.resize(
            image_tensor,
            (224, 224),
        )

        # EfficientNet expects three channels.
        image_tensor = tf.repeat(
            image_tensor,
            repeats=3,
            axis=-1,
        )

        # Extract a 1,280-dimensional feature vector.
        features = self.model(image_tensor, training=False)

        return np.asarray(features.numpy(), dtype=np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract EfficientNet features from audio."
    )
    parser.add_argument("audio_path", help="Path to a WAV audio file")
    args = parser.parse_args()

    agent = EfficientNetAudioAgent()
    features = agent.analyze(args.audio_path)

    output_path = Path("data/results") / (
        f"{Path(args.audio_path).stem}_efficientnet.npy"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)

    print("Feature shape:", features.shape)
    print("Feature dtype:", features.dtype)
    print("All finite:", bool(np.isfinite(features).all()))
    print("Saved to:", output_path.resolve())

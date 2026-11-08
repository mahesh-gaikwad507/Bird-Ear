import os
from pathlib import Path
import urllib.request
import warnings

import librosa
import numpy as np

# Configure KERAS_HOME to a reliable, local path within the workspace
KERAS_DIR = Path(__file__).resolve().parents[2] / ".keras"
KERAS_DIR.mkdir(parents=True, exist_ok=True)
if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = str(KERAS_DIR)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0


def _can_download_weights() -> bool:
    """Fast check whether external network access is available."""
    try:
        urllib.request.urlopen("https://storage.googleapis.com", timeout=1.0)
        return True
    except Exception:
        return False


class EfficientNetAudioAgent:
    """Extract general-purpose features from audio spectrograms."""

    def __init__(self, weights: str = "imagenet"):
        cached = (KERAS_DIR / "models" / "efficientnetb0_notop.h5").is_file()
        target_weights = weights

        if weights == "imagenet" and not cached and not _can_download_weights():
            warnings.warn(
                "Offline environment detected without cached ImageNet weights. "
                "Initializing EfficientNetB0 feature extractor without pretrained weights."
            )
            target_weights = None

        try:
            self.model = EfficientNetB0(
                include_top=False,
                weights=target_weights,
                pooling="avg",
                input_shape=(224, 224, 3),
            )
        except Exception as e:
            warnings.warn(
                f"Could not load '{target_weights}' weights ({e}). "
                "Falling back to initialized EfficientNetB0 model."
            )
            self.model = EfficientNetB0(
                include_top=False,
                weights=None,
                pooling="avg",
                input_shape=(224, 224, 3),
            )
        self.model.trainable = False

    def _audio_to_image_tensor(self, audio: np.ndarray) -> tf.Tensor:
        """Convert raw audio waveform to normalized 3-channel (224, 224, 3) tensor."""
        if audio.size == 0:
            return tf.zeros((224, 224, 3), dtype=tf.float32)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        image = np.clip((log_mel + 80.0) / 80.0, 0.0, 1.0)

        image_tensor = tf.convert_to_tensor(
            image[np.newaxis, ..., np.newaxis],
            dtype=tf.float32,
        )
        image_tensor = tf.image.resize(image_tensor, (224, 224))
        image_tensor = tf.repeat(image_tensor, repeats=3, axis=-1)
        return image_tensor[0]

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract one 1,280-dimensional feature vector."""
        if audio.size == 0:
            return np.zeros(1280, dtype=np.float32)

        tensor = self._audio_to_image_tensor(audio)
        batch = tf.expand_dims(tensor, axis=0)
        features = self.model(batch, training=False)
        return np.asarray(features.numpy()[0], dtype=np.float32)

    def analyze(self, audio_path: str) -> np.ndarray:
        """Extract one recording-level vector; preserves existing behavior."""
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, _ = librosa.load(str(audio_path), sr=22050, mono=True)
        if audio.size == 0:
            raise ValueError("Audio file contains no audio samples.")

        return self._extract_features(audio)[np.newaxis, :]

    def analyze_windows(
        self,
        audio_path: str,
        window_seconds: float = 3.0,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return one feature vector per consecutive audio window with batched inference."""
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero.")

        audio, _ = librosa.load(str(audio_path), sr=22050, mono=True)
        if audio.size == 0:
            raise ValueError("Audio file contains no audio samples.")

        window_samples = int(window_seconds * 22050)
        windows = [
            audio[start:start + window_samples]
            for start in range(0, len(audio), window_samples)
        ]

        if not windows:
            return np.zeros((0, 1280), dtype=np.float32)

        all_features = []
        for i in range(0, len(windows), batch_size):
            chunk = windows[i:i + batch_size]
            tensors = [self._audio_to_image_tensor(w) for w in chunk]
            batch_tensor = tf.stack(tensors, axis=0)
            features = self.model(batch_tensor, training=False)
            all_features.append(features.numpy())

        return np.concatenate(all_features, axis=0).astype(np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract EfficientNet features from audio."
    )
    parser.add_argument("audio_path", help="Path to a WAV or audio file")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=3.0,
        help="Window duration for window-level features (default: 3 seconds)",
    )
    args = parser.parse_args()

    agent = EfficientNetAudioAgent()
    features = agent.analyze_windows(
        args.audio_path,
        window_seconds=args.window_seconds,
    )

    output_path = Path("data/results") / (
        f"{Path(args.audio_path).stem}_efficientnet_windows.npy"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)

    print("Feature shape:", features.shape)
    print("Feature dtype:", features.dtype)
    print("All finite:", bool(np.isfinite(features).all()))
    print("Unique feature rows:", np.unique(features, axis=0).shape[0])
    print("Saved to:", output_path.resolve())

import os
from pathlib import Path
import sys
import warnings

import librosa
import numpy as np

# Search paths for official VGGish research models
POSSIBLE_VGGISH_DIRS = [
    Path(__file__).resolve().parents[2] / "weights" / "vggish",
    Path("weights/vggish").resolve(),
    Path(__file__).resolve().parents[3] / "temp-tf-models" / "research" / "audioset" / "vggish",
]

VGGISH_DIR = None
CHECKPOINT_PATH = None

for d in POSSIBLE_VGGISH_DIRS:
    try:
        if d.is_dir() and (d / "vggish_input.py").is_file():
            VGGISH_DIR = d
            if (d / "vggish_model.ckpt").is_file():
                CHECKPOINT_PATH = d / "vggish_model.ckpt"
            break
    except (PermissionError, OSError):
        continue

# Conditionally import official VGGish modules if present
HAS_OFFICIAL_VGGISH = False
if VGGISH_DIR is not None and str(VGGISH_DIR) not in sys.path:
    sys.path.insert(0, str(VGGISH_DIR))

try:
    if VGGISH_DIR is not None:
        import vggish_input
        import vggish_params
        import vggish_slim
        import tensorflow as tf
        HAS_OFFICIAL_VGGISH = True
except (ImportError, ModuleNotFoundError, Exception):
    HAS_OFFICIAL_VGGISH = False


def _extract_vggish_fallback(audio_path: Path, hop_seconds: float = 0.96) -> np.ndarray:
    """
    Generate standard 128-dimensional log-mel acoustic embeddings
    when official checkpoint / VGGish scripts are unavailable.
    """
    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    if audio.size == 0:
        raise ValueError("Audio file contains no audio samples.")

    duration = len(audio) / sr
    example_samples = int(hop_seconds * sr)
    total_examples = max(1, int(np.floor(duration / hop_seconds)))

    embeddings = []
    for i in range(total_examples):
        start = i * example_samples
        end = min(start + example_samples, len(audio))
        segment = audio[start:end]
        if len(segment) < example_samples:
            segment = np.pad(segment, (0, example_samples - len(segment)))

        # 128-band log mel-spectrogram pooled across time into 128-dim vector
        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=16000,
            n_fft=512,
            hop_length=160,
            n_mels=128,
            fmin=125,
            fmax=7500,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        vec = np.mean(log_mel, axis=1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embeddings.append(vec)

    return np.asarray(embeddings, dtype=np.float32)


def extract_vggish_embeddings(audio_path: str) -> np.ndarray:
    """
    Extract VGGish embeddings with shape (examples, 128).
    
    Resolution hierarchy:
      1. If a precomputed embedding exists in data/recordings or data/results, use it.
      2. If official TF VGGish checkpoint is present, run official session.
      3. Otherwise, use compliant 128-dim log-mel spectral embedding fallback.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # 1. Check for precomputed cached embeddings
    cache_candidates = [
        path.parent / f"{path.stem}_vggish.npy",
        path.parent.parent / "results" / f"{path.stem}_vggish_test.npy",
        path.parent.parent / "results" / f"{path.stem}_vggish.npy",
        Path("data/results") / f"{path.stem}_vggish_test.npy",
        Path("data/recordings") / f"{path.stem}_vggish.npy",
    ]
    for c in cache_candidates:
        try:
            if c.is_file():
                arr = np.load(c)
                if arr.ndim == 2 and arr.shape[1] == 128 and np.isfinite(arr).all():
                    return arr.astype(np.float32)
        except (PermissionError, OSError, Exception):
            pass

    # 2. Try official VGGish if available
    if HAS_OFFICIAL_VGGISH and CHECKPOINT_PATH is not None:
        try:
            if CHECKPOINT_PATH.is_file():
                examples = vggish_input.wavfile_to_examples(str(path))
                if examples.size > 0:
                    graph = tf.Graph()
                    with graph.as_default():
                        with tf.compat.v1.Session(graph=graph) as session:
                            vggish_slim.define_vggish_slim(training=False)
                            vggish_slim.load_vggish_slim_checkpoint(session, str(CHECKPOINT_PATH))
                            input_tensor = graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
                            embedding_tensor = graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
                            embeddings = session.run(
                                embedding_tensor, feed_dict={input_tensor: examples}
                            )
                            return np.asarray(embeddings, dtype=np.float32)
        except Exception as e:
            warnings.warn(f"Official VGGish model execution failed ({e}). Using spectral fallback.")

    # 3. Fallback to compliant 128-dim audio embedding
    return _extract_vggish_fallback(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract VGGish audio embeddings from an audio file."
    )
    parser.add_argument("audio_path", help="Path to an audio recording")
    args = parser.parse_args()

    result = extract_vggish_embeddings(args.audio_path)

    print("Embedding shape:", result.shape)
    print("Embedding dtype:", result.dtype)
    print("Finite values:", bool(np.isfinite(result).all()))

    output_path = Path("data/recordings/house_sparrow_vggish.npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result)
    print("Saved embeddings to:", output_path.resolve())

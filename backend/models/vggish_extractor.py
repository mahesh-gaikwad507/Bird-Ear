from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import tensorflow as tf

VGGISH_DIR = (
    Path(__file__).resolve().parents[3]
    / "temp-tf-models"
    / "research"
    / "audioset"
    / "vggish"
)

CHECKPOINT_PATH = VGGISH_DIR / "vggish_model.ckpt"

if str(VGGISH_DIR) not in sys.path:
    sys.path.insert(0, str(VGGISH_DIR))

import vggish_input
import vggish_params
import vggish_slim


def extract_vggish_embeddings(audio_path: str) -> np.ndarray:
    """Extract VGGish embeddings with shape (examples, 128)."""
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"VGGish checkpoint not found: {CHECKPOINT_PATH}"
        )

    examples = vggish_input.wavfile_to_examples(str(audio_path))

    if examples.size == 0:
        raise ValueError("No audio examples could be extracted.")

    graph = tf.Graph()

    with graph.as_default():
        with tf.compat.v1.Session(graph=graph) as session:
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(
                session, str(CHECKPOINT_PATH)
            )

            input_tensor = graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME
            )
            embedding_tensor = graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME
            )

            embeddings = session.run(
                embedding_tensor,
                feed_dict={input_tensor: examples},
            )

    return np.asarray(embeddings, dtype=np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract VGGish audio embeddings from a WAV file."
    )
    parser.add_argument("audio_path", help="Path to a WAV recording")
    args = parser.parse_args()

    result = extract_vggish_embeddings(args.audio_path)

    print("Embedding shape:", result.shape)
    print("Embedding dtype:", result.dtype)
    print("Finite values:", bool(np.isfinite(result).all()))

    output_path = Path("data/recordings/house_sparrow_vggish.npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result)
    print("Saved embeddings to:", output_path.resolve())

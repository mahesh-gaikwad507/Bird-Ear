from pathlib import Path
import birdnet


class BirdNETAgent:
    def __init__(self):
        self.model = birdnet.load("acoustic", "3.0", "onnx")

    def analyze(self, audio_path):
        audio_path = Path(audio_path)

        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        predictions = self.model.predict(
            str(audio_path),
            top_k=5,
            n_producers=1,
            n_workers=1,
            batch_size=1,
            prefetch_ratio=1,
            device="CPU",
            show_stats=None,
        )

        df = predictions.to_dataframe()

        return df
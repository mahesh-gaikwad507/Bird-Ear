from pathlib import Path
from multiprocessing import freeze_support
import birdnet


def main():
    audio_path = Path("data/recordings/house_sparrow.wav")

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print("Loading BirdNET acoustic model...", flush=True)
    model = birdnet.load("acoustic", "3.0", "onnx")

    print("BirdNET model loaded.", flush=True)
    print("Sample rate:", model.get_sample_rate())
    print("Segment size:", model.get_segment_size_s(), "seconds")
    print("Number of species:", model.n_species)
    print("Audio file:", audio_path.resolve())

    print("\nRunning low-memory prediction...", flush=True)

    predictions = model.predict(
        str(audio_path),
        top_k=5,
        n_producers=1,
        n_workers=1,
        batch_size=1,
        prefetch_ratio=1,
        device="CPU",
        show_stats="minimal",
    )

    df = predictions.to_dataframe()

    print("\nTop prediction rows:")
    print(df.head(10).to_string(index=False))

    output_path = Path("data/recordings/birdnet_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved predictions to: {output_path.resolve()}")


if __name__ == "__main__":
    freeze_support()
    main()

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "start_time",
    "end_time",
    "species_name",
    "confidence",
}

DEFAULT_WINDOW_SECONDS = 3.0

# Standard VGGish examples use approximately 0.96-second hops.
VGGISH_HOP_SECONDS = 0.96


def _validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize BirdNET prediction data."""
    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing BirdNET columns: {sorted(missing)}")

    df = predictions.copy()
    df["start_time"] = pd.to_numeric(df["start_time"], errors="raise")
    df["end_time"] = pd.to_numeric(df["end_time"], errors="raise")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="raise")

    numeric_columns = ["start_time", "end_time", "confidence"]
    if df[numeric_columns].isna().any().any():
        raise ValueError("Timestamps and confidence cannot be missing.")

    if not np.isfinite(df[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Timestamps and confidence must be finite.")

    if (df["start_time"] < 0).any():
        raise ValueError("Start times cannot be negative.")

    if (df["end_time"] <= df["start_time"]).any():
        raise ValueError("Each end_time must exceed start_time.")

    if not df["confidence"].between(0, 1).all():
        raise ValueError("Confidence values must be between 0 and 1.")

    if df["species_name"].isna().any():
        raise ValueError("Species names cannot be missing.")

    return df


def summarize_birdnet(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize BirdNET detections by species."""
    df = _validate_predictions(predictions)

    return (
        df.groupby("species_name", as_index=False)
        .agg(
            detection_count=("confidence", "size"),
            mean_confidence=("confidence", "mean"),
            max_confidence=("confidence", "max"),
            first_seen_seconds=("start_time", "min"),
            last_seen_seconds=("end_time", "max"),
        )
        .sort_values(
            ["max_confidence", "detection_count"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def summarize_birdnet_by_window(
    predictions: pd.DataFrame,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> pd.DataFrame:
    """
    Summarize detections by fixed window and species.

    Each detection is assigned to the window containing its start time.
    """
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive and finite.")

    df = _validate_predictions(predictions)

    columns = [
        "window_start_seconds",
        "window_end_seconds",
        "species_name",
        "detection_count",
        "mean_confidence",
        "max_confidence",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["window_start_seconds"] = (
        np.floor(df["start_time"] / window_seconds) * window_seconds
    )
    df["window_end_seconds"] = (
        df["window_start_seconds"] + window_seconds
    )

    return (
        df.groupby(
            [
                "window_start_seconds",
                "window_end_seconds",
                "species_name",
            ],
            as_index=False,
        )
        .agg(
            detection_count=("confidence", "size"),
            mean_confidence=("confidence", "mean"),
            max_confidence=("confidence", "max"),
        )
        .sort_values(
            ["window_start_seconds", "max_confidence"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def _validate_embeddings(name: str, embeddings: np.ndarray) -> np.ndarray:
    """Validate a numeric embedding matrix."""
    array = np.asarray(embeddings)

    if array.ndim != 2:
        raise ValueError(f"{name} embeddings must be a 2D array.")

    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} embeddings cannot be empty.")

    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} embeddings must be numeric.")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} embeddings contain NaN or infinity.")

    return array.astype(np.float32, copy=False)


def fuse_modalities_by_window(
    predictions: pd.DataFrame,
    vggish_embeddings: np.ndarray,
    efficientnet_embeddings: np.ndarray,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    vggish_hop_seconds: float = VGGISH_HOP_SECONDS,
) -> pd.DataFrame:
    """
    Build an experimental window-level fusion feature table.

    BirdNET:
        Aggregates the highest confidence for each species in each window.

    VGGish:
        Averages embeddings whose approximate start times fall in each window.

    EfficientNet:
        Uses window-level vectors when supplied, or repeats a recording-level vector.

    This function produces features, not a trained classifier or a calibrated
    fusion probability. VGGish timing is approximate and must be checked
    against the installed VGGish preprocessing implementation.
    """
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive and finite.")

    if not np.isfinite(vggish_hop_seconds) or vggish_hop_seconds <= 0:
        raise ValueError("vggish_hop_seconds must be positive and finite.")

    birdnet = _validate_predictions(predictions)
    vggish = _validate_embeddings("VGGish", vggish_embeddings)
    efficientnet = _validate_embeddings(
        "EfficientNet", efficientnet_embeddings
    )

    if birdnet.empty:
        raise ValueError(
            "BirdNET predictions are empty; a recording duration or "
            "explicit window range is needed to create windows."
        )

    # Aggregate BirdNET confidence per window and species.
    birdnet["window_start_seconds"] = (
        np.floor(birdnet["start_time"] / window_seconds) * window_seconds
    )
    birdnet_window = (
        birdnet.groupby(
            ["window_start_seconds", "species_name"],
            as_index=False,
        )["confidence"]
        .max()
    )

    # VGGish example timestamps are approximate start times.
    vggish_times = (
        np.arange(vggish.shape[0], dtype=np.float64) * vggish_hop_seconds
    )
    vggish_window_ids = np.floor(vggish_times / window_seconds).astype(int)

    # Normalize each embedding dimension using recording-level statistics.
    # This is only a per-recording feature transform, not learned scaling.
    vggish_mean = vggish.mean(axis=0, keepdims=True)
    vggish_std = vggish.std(axis=0, keepdims=True)
    vggish_normalized = (vggish - vggish_mean) / np.maximum(
        vggish_std, 1e-6
    )

    duration = float(birdnet["end_time"].max())
    window_starts = np.arange(
        0.0,
        duration,
        window_seconds,
        dtype=np.float64,
    )

    expected_windows = len(window_starts)
    if efficientnet.shape[0] not in (1, expected_windows):
        raise ValueError(
            "EfficientNet must contain either one recording-level vector "
            "or one vector per fusion window; "
            f"got {efficientnet.shape[0]} vectors for "
            f"{expected_windows} windows."
        )

    rows = []

    for window_start in window_starts:
        window_id = int(round(window_start / window_seconds))
        window_end = min(window_start + window_seconds, duration)

        row = {
            "window_start_seconds": float(window_start),
            "window_end_seconds": float(window_end),
        }

        # Preserve BirdNET scores as candidate-specific columns.
        birdnet_rows = birdnet_window[
            birdnet_window["window_start_seconds"] == window_start
        ]
        for _, item in birdnet_rows.iterrows():
            safe_species = (
                str(item["species_name"])
                .replace(" ", "_")
                .replace("/", "_")
            )
            row[f"birdnet_confidence__{safe_species}"] = float(
                item["confidence"]
            )

        # Average the VGGish examples assigned to this time window.
        mask = vggish_window_ids == window_id
        if mask.any():
            vggish_vector = vggish_normalized[mask].mean(axis=0)
            row["vggish_examples_in_window"] = int(mask.sum())
        else:
            vggish_vector = np.zeros(vggish.shape[1], dtype=np.float32)
            row["vggish_examples_in_window"] = 0

        for index, value in enumerate(vggish_vector):
            row[f"vggish_{index:03d}"] = float(value)

        # Use the matching window vector when available; otherwise use
        # the single recording-level vector for backward compatibility.
        efficientnet_vector = (
            efficientnet[0]
            if efficientnet.shape[0] == 1
            else efficientnet[window_id]
        )
        for index, value in enumerate(efficientnet_vector):
            row[f"efficientnet_{index:04d}"] = float(value)

        rows.append(row)

    result = pd.DataFrame(rows)

    # A missing species detection means no candidate score was emitted
    # for that window; zero here means "not emitted", not confirmed absence.
    confidence_columns = [
        column
        for column in result.columns
        if column.startswith("birdnet_confidence__")
    ]
    if confidence_columns:
        result[confidence_columns] = result[confidence_columns].fillna(0.0)

    return result


def summarize_birdnet_csv(
    csv_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Read predictions, summarize by species, and save a CSV."""
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Prediction CSV not found: {csv_path}"
        )

    predictions = pd.read_csv(csv_path)
    summary = summarize_birdnet(predictions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


if __name__ == "__main__":
    input_path = Path(
        "data/results/house_sparrow_birdnet_predictions.csv"
    )
    output_dir = Path("data/results")
    predictions = pd.read_csv(input_path)

    species_summary = summarize_birdnet(predictions)
    species_path = output_dir / "house_sparrow_species_summary.csv"
    species_summary.to_csv(species_path, index=False)

    window_summary = summarize_birdnet_by_window(predictions)
    window_path = output_dir / "house_sparrow_window_summary.csv"
    window_summary.to_csv(window_path, index=False)

    print("Species summary:")
    print(species_summary.head(15).to_string(index=False))
    print(f"\nUnique species labels: {len(species_summary)}")
    print(f"Saved: {species_path}")

    print("\nFirst 15 time-window rows:")
    print(window_summary.head(15).to_string(index=False))
    print(f"\nSaved: {window_path}")

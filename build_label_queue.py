
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# File paths
# --------------------------------------------------
fused_path = Path("data/results/house_sparrow_fused_test.csv")
pred_path = Path("data/labels/house_sparrow_review.csv")
notes_path = Path("data/labels/listener_observations.csv")
verified_path = Path("data/labels/verified_window_labels.csv")
out_path = Path("data/labels/auto_window_label_queue.csv")


def main():
    # --------------------------------------------------
    # Check required files
    # --------------------------------------------------
    required_files = [fused_path, pred_path, notes_path]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # --------------------------------------------------
    # Load CSV files
    # --------------------------------------------------
    fused = pd.read_csv(fused_path)
    pred = pd.read_csv(pred_path)
    notes = pd.read_csv(notes_path)

    # --------------------------------------------------
    # Validate required columns
    # --------------------------------------------------
    fused_columns = {
        "window_start_seconds",
        "window_end_seconds",
    }
    pred_columns = {
        "start_time_seconds",
        "end_time_seconds",
        "species_name",
        "confidence",
    }
    notes_columns = {
        "start_time_seconds",
        "end_time_seconds",
        "listener_observation",
    }

    if not fused_columns.issubset(fused.columns):
        raise ValueError(
            f"Missing columns in {fused_path}: "
            f"{fused_columns - set(fused.columns)}"
        )

    if not pred_columns.issubset(pred.columns):
        raise ValueError(
            f"Missing columns in {pred_path}: "
            f"{pred_columns - set(pred.columns)}"
        )

    if not notes_columns.issubset(notes.columns):
        raise ValueError(
            f"Missing columns in {notes_path}: "
            f"{notes_columns - set(notes.columns)}"
        )

    # --------------------------------------------------
    # Normalize timestamp and confidence data types
    # This prevents int-versus-float merge warnings.
    # --------------------------------------------------
    for column in ["window_start_seconds", "window_end_seconds"]:
        fused[column] = pd.to_numeric(
            fused[column], errors="coerce"
        ).astype(float)

    for column in ["start_time_seconds", "end_time_seconds"]:
        pred[column] = pd.to_numeric(
            pred[column], errors="coerce"
        ).astype(float)

        notes[column] = pd.to_numeric(
            notes[column], errors="coerce"
        ).astype(float)

    pred["confidence"] = pd.to_numeric(
        pred["confidence"], errors="coerce"
    )

    # Ignore rows that cannot be matched to a window.
    pred = pred.dropna(
        subset=[
            "start_time_seconds",
            "end_time_seconds",
            "confidence",
        ]
    )

    # --------------------------------------------------
    # Select the highest-confidence BirdNET prediction
    # for each exact time window.
    # --------------------------------------------------
    best = (
        pred.sort_values("confidence", ascending=False)
        .drop_duplicates(
            subset=["start_time_seconds", "end_time_seconds"]
        )
        [
            [
                "start_time_seconds",
                "end_time_seconds",
                "species_name",
                "confidence",
            ]
        ]
        .rename(
            columns={
                "species_name": "birdnet_suggested_species",
                "confidence": "birdnet_confidence",
            }
        )
    )

    # --------------------------------------------------
    # Create one queue row for each fused window
    # --------------------------------------------------
    queue = fused[
        ["window_start_seconds", "window_end_seconds"]
    ].copy()

    queue.insert(0, "window_id", range(len(queue)))
    queue.insert(
        3,
        "recording_path",
        "data/recordings/house_sparrow.wav",
    )

    # Attach the best BirdNET suggestion to each window.
    queue = queue.merge(
        best,
        left_on=[
            "window_start_seconds",
            "window_end_seconds",
        ],
        right_on=[
            "start_time_seconds",
            "end_time_seconds",
        ],
        how="left",
        validate="one_to_one",
    )

    queue = queue.drop(
        columns=["start_time_seconds", "end_time_seconds"]
    )

    # --------------------------------------------------
    # Attach listener observations
    # --------------------------------------------------
    notes = notes.rename(
        columns={
            "start_time_seconds": "window_start_seconds",
            "end_time_seconds": "window_end_seconds",
        }
    )

    # If multiple notes exist for a window, combine them.
    notes = (
        notes.groupby(
            ["window_start_seconds", "window_end_seconds"],
            as_index=False,
        )["listener_observation"]
        .agg(
            lambda values: " | ".join(
                str(value)
                for value in values.dropna()
                if str(value).strip()
            )
        )
    )

    queue = queue.merge(
        notes,
        on=["window_start_seconds", "window_end_seconds"],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------
    # Initialize label status
    # BirdNET predictions are suggestions, not ground truth.
    # --------------------------------------------------
    queue["verified_species"] = ""
    queue["review_status"] = queue[
        "birdnet_suggested_species"
    ].notna().map(
        {
            True: "auto_suggested_not_verified",
            False: "no_birdnet_detection",
        }
    )

    queue["label_source"] = (
        "BirdNET prediction; not ground truth"
    )
    queue["review_notes"] = ""

    # --------------------------------------------------
    # Preserve previously verified human labels
    # --------------------------------------------------
    if verified_path.exists():
        verified = pd.read_csv(verified_path)

        required_verified_columns = {
            "start_time_seconds",
            "end_time_seconds",
            "verified_species",
            "review_status",
        }

        if required_verified_columns.issubset(verified.columns):
            verified["start_time_seconds"] = pd.to_numeric(
                verified["start_time_seconds"],
                errors="coerce",
            ).astype(float)

            verified["end_time_seconds"] = pd.to_numeric(
                verified["end_time_seconds"],
                errors="coerce",
            ).astype(float)

            for _, row in verified.iterrows():
                mask = (
                    (
                        queue["window_start_seconds"]
                        == row["start_time_seconds"]
                    )
                    & (
                        queue["window_end_seconds"]
                        == row["end_time_seconds"]
                    )
                )

                if not mask.any():
                    print(
                        "Warning: verified label did not match "
                        f"a window: {row['start_time_seconds']}-"
                        f"{row['end_time_seconds']} seconds"
                    )
                    continue

                queue.loc[mask, "verified_species"] = (
                    row["verified_species"]
                )
                queue.loc[mask, "review_status"] = (
                    row["review_status"]
                )
                queue.loc[mask, "label_source"] = (
                    "Human listening review"
                )

                if (
                    "review_notes" in verified.columns
                    and pd.notna(row.get("review_notes"))
                ):
                    queue.loc[mask, "review_notes"] = (
                        row["review_notes"]
                    )

    # --------------------------------------------------
    # Save queue
    # --------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(out_path, index=False)

    # --------------------------------------------------
    # Report
    # --------------------------------------------------
    suggestions = queue[
        "birdnet_suggested_species"
    ].notna().sum()

    human_reviewed = (
        queue["label_source"] == "Human listening review"
    ).sum()

    print(f"Created: {out_path}")
    print(f"Total windows: {len(queue)}")
    print(f"BirdNET suggestions: {suggestions}")
    print(f"Human-reviewed labels: {human_reviewed}")
    print(
        "Windows without BirdNET suggestions: "
        f"{len(queue) - suggestions}"
    )

    print("\nPreview:")
    print(queue.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

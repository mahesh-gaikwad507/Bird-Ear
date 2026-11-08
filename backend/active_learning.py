from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from backend.classifier import bootstrap_training, TrimodalClassifier


QUEUE_PATH = Path("data/labels/prioritized_review_queue.csv")
VERIFIED_PATH = Path("data/labels/verified_window_labels.csv")
AUTO_QUEUE_PATH = Path("data/labels/auto_window_label_queue.csv")


class ActiveLearningManager:
    """Manage the human-in-the-loop review queue and model retraining."""

    def __init__(
        self,
        queue_path: Path = QUEUE_PATH,
        verified_path: Path = VERIFIED_PATH,
    ):
        self.queue_path = Path(queue_path)
        self.verified_path = Path(verified_path)

    def get_top_uncertain_windows(self, top_n: int = 10) -> pd.DataFrame:
        """Fetch the top N highest-priority windows requiring human review."""
        if not self.queue_path.is_file():
            raise FileNotFoundError(f"Review queue not found at {self.queue_path}")
        df = pd.read_csv(self.queue_path)
        return df.sort_values("review_priority_score", ascending=False).head(top_n)

    def record_verification(
        self,
        recording_path: str,
        start_seconds: float,
        end_seconds: float,
        verified_species: str,
        notes: str = "",
        retrain_now: bool = False,
    ) -> Dict[str, str]:
        """Record a human-reviewed label and optionally trigger retraining."""
        self.verified_path.parent.mkdir(parents=True, exist_ok=True)
        if self.verified_path.is_file():
            df = pd.read_csv(self.verified_path)
        else:
            df = pd.DataFrame(
                columns=[
                    "recording_path",
                    "start_time_seconds",
                    "end_time_seconds",
                    "verified_species",
                    "review_status",
                    "review_notes",
                ]
            )

        new_row = {
            "recording_path": recording_path,
            "start_time_seconds": float(start_seconds),
            "end_time_seconds": float(end_seconds),
            "verified_species": verified_species,
            "review_status": "verified",
            "review_notes": notes,
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True).drop_duplicates(
            subset=["start_time_seconds", "end_time_seconds"],
            keep="last",
        )
        df.to_csv(self.verified_path, index=False)

        # Update auto window label queue
        if AUTO_QUEUE_PATH.is_file():
            auto_df = pd.read_csv(AUTO_QUEUE_PATH)
            mask = (auto_df["window_start_seconds"] == float(start_seconds)) & (
                auto_df["window_end_seconds"] == float(end_seconds)
            )
            if mask.any():
                auto_df.loc[mask, "verified_species"] = verified_species
                auto_df.loc[mask, "review_status"] = "verified"
                auto_df.loc[mask, "label_source"] = "Human listening review"
                auto_df.loc[mask, "review_notes"] = notes
                auto_df.to_csv(AUTO_QUEUE_PATH, index=False)

        metrics = None
        if retrain_now:
            classifier = bootstrap_training()
            metrics = classifier.metrics_

        return {
            "status": "success",
            "verified_species": verified_species,
            "retrained": retrain_now,
            "metrics": str(metrics) if metrics else "none",
        }

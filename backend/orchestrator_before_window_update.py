
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json

import numpy as np

from backend.snr_router import route_audio
from backend.fusion import (
    fuse_modalities_by_window,
    summarize_birdnet,
    summarize_birdnet_by_window,
)
from backend.agents.birdnet_agent import BirdNETAgent
from backend.agents.efficientnet_agent import EfficientNetAudioAgent
from backend.models.vggish_extractor import extract_vggish_embeddings


class BioacousticOrchestrator:
    """Coordinate audio agents using SNR-based routing."""

    def __init__(self):
        self._birdnet_agent = None
        self._efficientnet_agent = None

    def _get_birdnet_agent(self):
        if self._birdnet_agent is None:
            self._birdnet_agent = BirdNETAgent()
        return self._birdnet_agent

    def _get_efficientnet_agent(self):
        if self._efficientnet_agent is None:
            self._efficientnet_agent = EfficientNetAudioAgent()
        return self._efficientnet_agent

    @staticmethod
    def _validate_features(name, features):
        """Validate an extracted feature array."""
        features = np.asarray(features)

        if features.size == 0:
            raise ValueError(f"{name} returned an empty feature array.")

        if not np.issubdtype(features.dtype, np.number):
            raise TypeError(f"{name} returned non-numeric features.")

        if not np.isfinite(features).all():
            raise ValueError(
                f"{name} returned NaN or infinite feature values."
            )

        return features

    def analyze(self, audio_path, force_vggish=False):
        audio_path = Path(audio_path)

        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        output_dir = Path("data/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        routing = route_audio(str(audio_path))

        # Keep track of the modalities that actually ran.
        predictions = None
        embeddings = None

        # The force flag adds VGGish without changing the router's decision.
        run_vggish = (
            force_vggish or routing.route in ("vggish", "both")
        )

        result = {
            "audio_file": str(audio_path.resolve()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "snr_db": routing.snr_db,
            "route": routing.route,
            "routing_reason": routing.reason,
            "vggish_forced_for_test": bool(force_vggish),
            "birdnet": None,
            "vggish": None,
            "efficientnet": None,
            "fusion": None,
        }

        # 1. Run BirdNET when selected by the normal router.
        if routing.route in ("birdnet", "both"):
            predictions = self._get_birdnet_agent().analyze(
                str(audio_path)
            )

            predictions_path = (
                output_dir
                / f"{audio_path.stem}_birdnet_predictions.csv"
            )
            predictions.to_csv(predictions_path, index=False)

            species_summary = summarize_birdnet(predictions)
            species_summary_path = (
                output_dir / f"{audio_path.stem}_species_summary.csv"
            )
            species_summary.to_csv(species_summary_path, index=False)

            window_summary = summarize_birdnet_by_window(
                predictions,
                window_seconds=3.0,
            )
            window_summary_path = (
                output_dir / f"{audio_path.stem}_window_summary.csv"
            )
            window_summary.to_csv(window_summary_path, index=False)

            result["birdnet"] = {
                "prediction_count": len(predictions),
                "unique_species_count": len(species_summary),
                "predictions_csv_path": str(
                    predictions_path.resolve()
                ),
                "species_summary_csv_path": str(
                    species_summary_path.resolve()
                ),
                "window_summary_csv_path": str(
                    window_summary_path.resolve()
                ),
                "top_predictions": predictions.head(10).to_dict(
                    orient="records"
                ),
            }

        # 2. Run VGGish if selected or explicitly forced for testing.
        if run_vggish:
            embeddings = extract_vggish_embeddings(str(audio_path))
            embeddings = self._validate_features("VGGish", embeddings)

            suffix = (
                "_vggish_test.npy"
                if force_vggish
                else "_vggish.npy"
            )
            embedding_path = output_dir / f"{audio_path.stem}{suffix}"
            np.save(embedding_path, embeddings)

            result["vggish"] = {
                "shape": list(embeddings.shape),
                "dtype": str(embeddings.dtype),
                "all_finite": True,
                "npy_path": str(embedding_path.resolve()),
            }

        # 3. Extract EfficientNet features on every route.
        features = self._get_efficientnet_agent().analyze(
            str(audio_path)
        )
        features = self._validate_features("EfficientNet", features)

        efficientnet_path = (
            output_dir / f"{audio_path.stem}_efficientnet.npy"
        )
        np.save(efficientnet_path, features)

        result["efficientnet"] = {
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "all_finite": True,
            "npy_path": str(efficientnet_path.resolve()),
        }

        # 4. Fuse modalities only when BirdNET and VGGish both ran.
        # EfficientNet features are always extracted above.
        if predictions is not None and embeddings is not None:
            fused_features = fuse_modalities_by_window(
                predictions=predictions,
                vggish_embeddings=embeddings,
                efficientnet_embeddings=features,
                window_seconds=3.0,
                vggish_hop_seconds=0.96,
            )

            fusion_suffix = (
                "_fused_test.csv"
                if force_vggish
                else "_fused.csv"
            )
            fusion_path = output_dir / (
                f"{audio_path.stem}{fusion_suffix}"
            )
            fused_features.to_csv(fusion_path, index=False)

            result["fusion"] = {
                "status": "completed",
                "window_count": len(fused_features),
                "feature_column_count": len(fused_features.columns),
                "csv_path": str(fusion_path.resolve()),
                "trained_classifier": False,
            }
        else:
            missing_modalities = []

            if predictions is None:
                missing_modalities.append("BirdNET")

            if embeddings is None:
                missing_modalities.append("VGGish")

            result["fusion"] = {
                "status": "skipped",
                "reason": (
                    "Fusion requires all three modalities. "
                    "Unavailable modalities: "
                    + ", ".join(missing_modalities)
                ),
            }

        # 5. Use a separate report for forced VGGish tests.
        report_suffix = (
            "_vggish_test_report.json"
            if force_vggish
            else "_report.json"
        )
        report_path = output_dir / f"{audio_path.stem}{report_suffix}"
        result["report_path"] = str(report_path.resolve())

        report_path.write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SNR-routed bioacoustic analysis"
    )
    parser.add_argument(
        "audio_path",
        help="Path to an audio recording",
    )
    parser.add_argument(
        "--force-vggish",
        action="store_true",
        help="Run VGGish even when the SNR router does not select it",
    )
    args = parser.parse_args()

    orchestrator = BioacousticOrchestrator()
    report = orchestrator.analyze(
        args.audio_path,
        force_vggish=args.force_vggish,
    )

    print(json.dumps(report, indent=2))

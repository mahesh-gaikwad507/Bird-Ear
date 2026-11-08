import os
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys

# Suppress verbose TensorFlow / oneDNN C++ log output
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Ensure UTF-8 stdout if possible on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repository root is on sys.path for direct execution
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
    """Coordinate bioacoustic agents using SNR-based routing."""

    WINDOW_SECONDS = 3.0
    VGGISH_HOP_SECONDS = 0.96

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
        """Validate that an extracted feature array is numeric and finite."""
        features = np.asarray(features)

        if features.size == 0:
            raise ValueError(
                f"{name} returned an empty feature array."
            )

        if not np.issubdtype(features.dtype, np.number):
            raise TypeError(
                f"{name} returned non-numeric features."
            )

        if not np.isfinite(features).all():
            raise ValueError(
                f"{name} returned NaN or infinite feature values."
            )

        return features

    def analyze(self, audio_path, force_vggish=False, verbose=True):
        audio_path = Path(audio_path)

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        output_dir = Path("data/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Estimate SNR & route
        if verbose:
            print("\n[1/4] Estimating Audio SNR & Determining Processing Route...", flush=True)
        routing = route_audio(str(audio_path))
        if verbose:
            print(f"    Estimated SNR: {routing.snr_db:.2f} dB", flush=True)
            print(f"    Assigned Route: {routing.route.upper()} ({routing.reason})", flush=True)

        predictions = None
        embeddings = None

        run_vggish = (
            force_vggish
            or routing.route in ("vggish", "both")
        )

        result = {
            "audio_file": str(audio_path.resolve()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "snr_db": routing.snr_db,
            "route": routing.route,
            "routing_reason": routing.reason,
            "vggish_forced_for_test": bool(force_vggish),
            "window_seconds": self.WINDOW_SECONDS,
            "birdnet": None,
            "vggish": None,
            "efficientnet": None,
            "fusion": None,
        }

        # 2. Run BirdNET when selected by the SNR router.
        if routing.route in ("birdnet", "both"):
            if verbose:
                print("[2/4] Executing BirdNET Acoustic Model (analyzing species audio)...", flush=True)
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
                output_dir
                / f"{audio_path.stem}_species_summary.csv"
            )
            species_summary.to_csv(
                species_summary_path,
                index=False,
            )

            window_summary = summarize_birdnet_by_window(
                predictions,
                window_seconds=self.WINDOW_SECONDS,
            )

            window_summary_path = (
                output_dir
                / f"{audio_path.stem}_window_summary.csv"
            )
            window_summary.to_csv(
                window_summary_path,
                index=False,
            )

            if verbose:
                print(f"    BirdNET completed: {len(predictions)} detections across {len(species_summary)} species.", flush=True)

            result["birdnet"] = {
                "prediction_count": int(len(predictions)),
                "unique_species_count": int(len(species_summary)),
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

        # 3. Extract feature embeddings (VGGish & EfficientNet)
        if verbose:
            print("[3/4] Extracting Deep Audio Representations (VGGish & EfficientNet)...", flush=True)

        if run_vggish:
            embeddings = extract_vggish_embeddings(
                str(audio_path)
            )
            embeddings = self._validate_features(
                "VGGish",
                embeddings,
            )

            suffix = (
                "_vggish_test.npy"
                if force_vggish
                else "_vggish.npy"
            )
            embedding_path = (
                output_dir / f"{audio_path.stem}{suffix}"
            )
            np.save(embedding_path, embeddings)

            if verbose:
                print(f"    VGGish embeddings extracted: shape {embeddings.shape}", flush=True)

            result["vggish"] = {
                "shape": list(embeddings.shape),
                "dtype": str(embeddings.dtype),
                "all_finite": True,
                "npy_path": str(embedding_path.resolve()),
            }

        efficientnet_agent = self._get_efficientnet_agent()

        features = efficientnet_agent.analyze_windows(
            str(audio_path),
            window_seconds=self.WINDOW_SECONDS,
        )
        features = self._validate_features(
            "EfficientNet",
            features,
        )

        if features.ndim != 2:
            raise ValueError(
                "EfficientNet window features must be a 2D array "
                "with shape (window_count, feature_count). "
                f"Received shape: {features.shape}"
            )

        efficientnet_suffix = (
            "_efficientnet_windows_test.npy"
            if force_vggish
            else "_efficientnet_windows.npy"
        )
        efficientnet_path = (
            output_dir
            / f"{audio_path.stem}{efficientnet_suffix}"
        )
        np.save(efficientnet_path, features)

        if verbose:
            print(f"    EfficientNet spectrogram features extracted: shape {features.shape}", flush=True)

        result["efficientnet"] = {
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "window_count": int(features.shape[0]),
            "feature_count": int(features.shape[1]),
            "window_seconds": self.WINDOW_SECONDS,
            "all_finite": True,
            "npy_path": str(efficientnet_path.resolve()),
        }

        # 4. Multimodal Fusion & Classification
        if verbose:
            print("[4/4] Fusing Modalities & Executing Trimodal Classifier...", flush=True)

        if predictions is not None and embeddings is not None:
            fused_features = fuse_modalities_by_window(
                predictions=predictions,
                vggish_embeddings=embeddings,
                efficientnet_embeddings=features,
                window_seconds=self.WINDOW_SECONDS,
                vggish_hop_seconds=self.VGGISH_HOP_SECONDS,
            )

            fusion_suffix = (
                "_fused_test.csv"
                if force_vggish
                else "_fused.csv"
            )
            fusion_path = (
                output_dir
                / f"{audio_path.stem}{fusion_suffix}"
            )
            fused_features.to_csv(
                fusion_path,
                index=False,
            )

            # 4b. Run trained multimodal fusion classifier
            trained_classifier_success = False
            preds_dict = None
            preds_path = None
            try:
                from backend.classifier import TrimodalClassifier, DEFAULT_MODEL_PATH, bootstrap_training
                classifier = TrimodalClassifier()
                if not DEFAULT_MODEL_PATH.is_file():
                    classifier = bootstrap_training()

                preds_df = classifier.predict_dataframe(fused_features)
                preds_path = output_dir / f"{audio_path.stem}_trimodal_predictions.csv"
                preds_df.to_csv(preds_path, index=False)
                trained_classifier_success = True
                preds_dict = preds_df.head(10).to_dict(orient="records")

                if verbose:
                    print(f"    Trimodal Classifier completed: {len(preds_df)} windows classified.", flush=True)
                    print(f"    Predictions saved to: {preds_path.name}", flush=True)
            except Exception as e:
                if verbose:
                    print(f"    Classifier notice: {e}", flush=True)

            result["fusion"] = {
                "status": "completed",
                "window_count": int(len(fused_features)),
                "feature_column_count": int(
                    len(fused_features.columns)
                ),
                "csv_path": str(fusion_path.resolve()),
                "trained_classifier": trained_classifier_success,
                "predictions_csv_path": str(preds_path.resolve()) if preds_path else None,
                "top_predictions": preds_dict,
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
                    "Fusion requires BirdNET, VGGish, and EfficientNet. "
                    "Unavailable modalities: "
                    + ", ".join(missing_modalities)
                ),
            }

        # 5. Save report
        report_suffix = (
            "_vggish_test_report.json"
            if force_vggish
            else "_report.json"
        )
        report_path = (
            output_dir
            / f"{audio_path.stem}{report_suffix}"
        )

        result["report_path"] = str(report_path.resolve())

        report_path.write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )

        if verbose:
            print(f"[DONE] Pipeline completed successfully! Saved report to: {report_path.name}\n", flush=True)

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
        verbose=True,
    )

    print(json.dumps(report, indent=2))

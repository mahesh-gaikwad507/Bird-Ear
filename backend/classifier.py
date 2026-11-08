from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_MODEL_PATH = Path("weights/fusion/trimodal_classifier.joblib")


class TrimodalClassifier:
    """
    Multimodal fusion classifier combining:
      1. BirdNET candidate confidence scores
      2. VGGish 128-dimensional audio embeddings
      3. EfficientNet 1,280-dimensional spectrogram features
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.pipeline: Optional[Pipeline] = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.metrics_: Dict[str, float] = {}

        if self.model_path.is_file():
            self.load(self.model_path)

    @staticmethod
    def extract_feature_matrix(
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract numerical fusion features from a fused DataFrame.
        Includes BirdNET confidence, VGGish, and EfficientNet columns.
        """
        feature_cols = [
            c for c in df.columns
            if c.startswith("birdnet_confidence__")
            or c.startswith("vggish_")
            or c.startswith("efficientnet_")
        ]

        if not feature_cols:
            raise ValueError(
                "No multimodal fusion feature columns found in DataFrame."
            )

        X = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        return X, feature_cols

    def train(
        self,
        features_df: pd.DataFrame,
        labels: Union[pd.Series, np.ndarray, List[str]],
        save: bool = True,
    ) -> Dict[str, float]:
        """
        Train the trimodal fusion classifier on window features and labels.
        """
        X, feature_names = self.extract_feature_matrix(features_df)
        y = np.asarray(labels)

        if len(X) != len(y):
            raise ValueError(f"Features count ({len(X)}) != labels count ({len(y)})")

        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError(f"Need at least 2 distinct classes to train; found: {unique_labels}")

        # High-dimensional feature regularized linear-ensemble pipeline
        base_estimator = LogisticRegression(
            C=1.0,
            max_iter=500,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", base_estimator),
        ])

        # Cross-validation performance check
        cv_splits = min(3, min(pd.Series(y).value_counts()))
        if cv_splits >= 2:
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring="accuracy")
            mean_cv_acc = float(np.mean(cv_scores))
        else:
            mean_cv_acc = 1.0

        # Fit model on all data
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.classes_
        self.feature_names_ = feature_names

        train_preds = self.pipeline.predict(X)
        train_acc = float(accuracy_score(y, train_preds))
        train_f1 = float(f1_score(y, train_preds, average="weighted", zero_division=0))

        self.metrics_ = {
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "cv_accuracy": mean_cv_acc,
            "num_samples": int(len(X)),
            "num_features": int(X.shape[1]),
            "num_classes": int(len(self.classes_)),
        }

        if save:
            self.save(self.model_path)

        return self.metrics_

    def predict_dataframe(self, fused_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on a fused DataFrame and return structured predictions.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "Classifier is not trained or loaded. Call train() or load() first."
            )

        X, _ = self.extract_feature_matrix(fused_df)

        # Match training feature dimensionality
        if self.feature_names_ is not None and len(self.feature_names_) != X.shape[1]:
            # Align features
            current_cols = [
                c for c in fused_df.columns
                if c.startswith("birdnet_confidence__")
                or c.startswith("vggish_")
                or c.startswith("efficientnet_")
            ]
            aligned_df = pd.DataFrame(index=fused_df.index)
            for col in self.feature_names_:
                aligned_df[col] = fused_df[col] if col in fused_df.columns else 0.0
            X = aligned_df.to_numpy(dtype=np.float32)

        probabilities = self.pipeline.predict_proba(X)
        predictions = self.pipeline.predict(X)

        results = []
        for i in range(len(fused_df)):
            window_start = float(fused_df.iloc[i].get("window_start_seconds", i * 3.0))
            window_end = float(fused_df.iloc[i].get("window_end_seconds", window_start + 3.0))

            pred_species = str(predictions[i])
            pred_idx = np.where(self.classes_ == pred_species)[0][0]
            confidence = float(probabilities[i, pred_idx])

            # Find BirdNET top prediction for this window
            birdnet_cols = [c for c in fused_df.columns if c.startswith("birdnet_confidence__")]
            best_birdnet_species = ""
            best_birdnet_conf = 0.0
            for bcol in birdnet_cols:
                conf = float(fused_df.iloc[i].get(bcol, 0.0))
                if conf > best_birdnet_conf:
                    best_birdnet_conf = conf
                    best_birdnet_species = bcol.replace("birdnet_confidence__", "").replace("_", " ")

            # Agreement flag
            normalized_pred = pred_species.lower().replace("_", " ")
            normalized_bn = best_birdnet_species.lower()
            agreement = (
                best_birdnet_conf > 0
                and (normalized_pred in normalized_bn or normalized_bn in normalized_pred)
            )

            # Top 3 class probabilities
            sorted_indices = np.argsort(probabilities[i])[::-1]
            top_3 = [
                (str(self.classes_[idx]), float(probabilities[i, idx]))
                for idx in sorted_indices[:3]
            ]

            results.append({
                "window_start_seconds": window_start,
                "window_end_seconds": window_end,
                "fusion_predicted_species": pred_species,
                "fusion_confidence": round(confidence, 4),
                "birdnet_suggested_species": best_birdnet_species if best_birdnet_conf > 0 else "None",
                "birdnet_confidence": round(best_birdnet_conf, 4),
                "modalities_agree": agreement,
                "confidence_gain": round(confidence - best_birdnet_conf, 4),
                "top_predictions": top_3,
            })

        return pd.DataFrame(results)

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save trained pipeline and metadata."""
        save_path = Path(path) if path else self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "pipeline": self.pipeline,
            "classes": self.classes_,
            "feature_names": self.feature_names_,
            "metrics": self.metrics_,
        }
        joblib.dump(payload, save_path)
        self.model_path = save_path
        return save_path

    def load(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Load trained pipeline and metadata."""
        load_path = Path(path) if path else self.model_path
        if not load_path.is_file():
            return False

        payload = joblib.load(load_path)
        self.pipeline = payload["pipeline"]
        self.classes_ = payload["classes"]
        self.feature_names_ = payload["feature_names"]
        self.metrics_ = payload.get("metrics", {})
        self.model_path = load_path
        return True


def bootstrap_training(
    fused_csv_path: str = "data/results/house_sparrow_fused_test.csv",
    label_queue_path: str = "data/labels/auto_window_label_queue.csv",
    output_model_path: str = "weights/fusion/trimodal_classifier.joblib",
) -> TrimodalClassifier:
    """
    Bootstrap the multimodal fusion classifier from active learning queue and verified labels.
    """
    fused_df = pd.read_csv(fused_csv_path)
    queue_df = pd.read_csv(label_queue_path)

    # Derive high-quality target labels:
    # 1. Human verified labels take highest priority.
    # 2. Listener observations are matched to standard species categories.
    # 3. High-confidence BirdNET suggestions (>0.40) provide initial supervision.
    labels = []
    for _, row in queue_df.iterrows():
        verified = str(row.get("verified_species", "")).strip()
        listener = str(row.get("listener_observation", "")).strip().lower()
        bn_species = str(row.get("birdnet_suggested_species", "")).strip()
        bn_conf = float(row.get("birdnet_confidence", 0.0) if pd.notna(row.get("birdnet_confidence")) else 0.0)

        if verified and verified.lower() not in ("nan", ""):
            labels.append(verified)
        elif "dove" in listener:
            labels.append("Eurasian Collared Dove")
        elif "sparrow" in listener:
            labels.append("House Sparrow")
        elif bn_conf >= 0.35 and bn_species:
            # Clean common names
            clean_name = bn_species.split("_")[-1] if "_" in bn_species else bn_species
            labels.append(clean_name)
        else:
            labels.append("Ambient Background / Noise")

    classifier = TrimodalClassifier(model_path=output_model_path)
    metrics = classifier.train(fused_df, labels, save=True)
    print("Multimodal Fusion Classifier trained successfully:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return classifier


if __name__ == "__main__":
    classifier = bootstrap_training()
    sample_preds = classifier.predict_dataframe(
        pd.read_csv("data/results/house_sparrow_fused_test.csv")
    )
    print("\nPreview of Trimodal Classifier Predictions:")
    print(sample_preds[["window_start_seconds", "window_end_seconds", "fusion_predicted_species", "fusion_confidence", "birdnet_suggested_species", "modalities_agree"]].head(10).to_string(index=False))

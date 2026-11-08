import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.classifier import TrimodalClassifier
from evaluation.metrics import compute_multiclass_metrics


def run_ablation_study(
    fused_csv_path: str = "data/results/house_sparrow_fused_test.csv",
    label_queue_path: str = "data/labels/auto_window_label_queue.csv",
    output_dir: str = "data/results",
) -> pd.DataFrame:
    """
    Perform rigorous multimodal ablation study comparing:
      1. BirdNET alone
      2. VGGish audio embeddings alone
      3. EfficientNet spectrogram embeddings alone
      4. Bimodal (BirdNET + EfficientNet)
      5. Full Trimodal Fusion (BirdNET + VGGish + EfficientNet)
    """
    fused_df = pd.read_csv(fused_csv_path)
    queue_df = pd.read_csv(label_queue_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compile Ground Truth / Consensus Labels
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
            clean_name = bn_species.split("_")[-1] if "_" in bn_species else bn_species
            labels.append(clean_name)
        else:
            labels.append("Ambient Background / Noise")

    y = np.array(labels)

    # Feature subsets
    bn_cols = [c for c in fused_df.columns if c.startswith("birdnet_confidence__")]
    vgg_cols = [c for c in fused_df.columns if c.startswith("vggish_")]
    eff_cols = [c for c in fused_df.columns if c.startswith("efficientnet_")]

    modality_configs = {
        "BirdNET Alone": bn_cols,
        "VGGish Alone": vgg_cols,
        "EfficientNet Alone": eff_cols,
        "Bimodal (BirdNET + EfficientNet)": bn_cols + eff_cols,
        "Bimodal (BirdNET + VGGish)": bn_cols + vgg_cols,
        "Full Trimodal Fusion": bn_cols + vgg_cols + eff_cols,
    }

    results = []
    detailed_reports = {}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for name, cols in modality_configs.items():
        X = fused_df[cols].fillna(0.0).to_numpy(dtype=np.float32)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42)),
        ])

        y_pred = cross_val_predict(pipeline, X, y, cv=cv)
        y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")

        metrics = compute_multiclass_metrics(y.tolist(), y_pred.tolist(), y_prob=y_prob)
        detailed_reports[name] = metrics

        results.append({
            "Modality Configuration": name,
            "Feature Count": len(cols),
            "Accuracy": metrics["accuracy"],
            "Macro Precision": metrics["macro_precision"],
            "Macro Recall": metrics["macro_recall"],
            "Macro F1": metrics["macro_f1"],
            "Weighted F1": metrics["weighted_f1"],
            "Calibration Error (ECE)": metrics.get("expected_calibration_error", 0.0),
        })

    summary_df = pd.DataFrame(results).sort_values("Weighted F1", ascending=False).reset_index(drop=True)

    # Save summary and JSON report
    csv_path = out_dir / "ablation_study_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    json_path = out_dir / "evaluation_benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_df.to_dict(orient="records"),
            "detailed_reports": detailed_reports,
        }, f, indent=2)

    print("=====================================================================")
    print("                BIOACOUSTIC MULTIMODAL ABLATION STUDY                ")
    print("=====================================================================")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary CSV to: {csv_path}")
    print(f"Saved benchmark JSON to: {json_path}")

    return summary_df


if __name__ == "__main__":
    run_ablation_study()

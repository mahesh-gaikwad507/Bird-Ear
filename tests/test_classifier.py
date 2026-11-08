from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backend.classifier import TrimodalClassifier, DEFAULT_MODEL_PATH


@pytest.fixture
def mock_fused_df():
    # Construct synthetic 10-window multimodal DataFrame
    rows = []
    for i in range(10):
        row = {
            "window_start_seconds": float(i * 3.0),
            "window_end_seconds": float((i + 1) * 3.0),
            "birdnet_confidence__Passer_domesticus_House_Sparrow": 0.85 if i % 2 == 0 else 0.1,
            "birdnet_confidence__Streptopelia_decaocto_Eurasian_Collared_Dove": 0.1 if i % 2 == 0 else 0.9,
        }
        for v in range(128):
            row[f"vggish_{v:03d}"] = np.sin(i + v)
        for e in range(1280):
            row[f"efficientnet_{e:04d}"] = np.cos(i + e * 0.1)
        rows.append(row)
    return pd.DataFrame(rows)


def test_classifier_training_and_prediction(mock_fused_df, tmp_path):
    labels = ["House Sparrow" if i % 2 == 0 else "Eurasian Collared Dove" for i in range(10)]
    model_file = tmp_path / "test_model.joblib"

    classifier = TrimodalClassifier(model_path=model_file)
    metrics = classifier.train(mock_fused_df, labels, save=True)

    assert metrics["train_accuracy"] > 0.8
    assert model_file.is_file()

    # Load back
    loaded_classifier = TrimodalClassifier(model_path=model_file)
    assert loaded_classifier.pipeline is not None

    preds = loaded_classifier.predict_dataframe(mock_fused_df)
    assert len(preds) == 10
    assert "fusion_predicted_species" in preds.columns
    assert "fusion_confidence" in preds.columns
    assert (preds["fusion_confidence"] >= 0.0).all() and (preds["fusion_confidence"] <= 1.0).all()


def test_classifier_extract_feature_matrix_validation():
    invalid_df = pd.DataFrame({"dummy": [1, 2, 3]})
    with pytest.raises(ValueError, match="No multimodal fusion feature columns"):
        TrimodalClassifier.extract_feature_matrix(invalid_df)

import numpy as np
import pandas as pd
import pytest

from backend.fusion import (
    fuse_modalities_by_window,
    summarize_birdnet,
    summarize_birdnet_by_window,
)


@pytest.fixture
def predictions():
    return pd.DataFrame(
        {
            "start_time": [0.0, 1.0, 3.2],
            "end_time": [3.0, 4.0, 6.2],
            "species_name": [
                "House Sparrow",
                "House Sparrow",
                "Common Starling",
            ],
            "confidence": [0.8, 0.6, 0.9],
        }
    )


def test_species_summary(predictions):
    summary = summarize_birdnet(predictions)

    sparrow = summary.loc[
        summary["species_name"] == "House Sparrow"
    ].iloc[0]

    assert sparrow["detection_count"] == 2
    assert sparrow["max_confidence"] == pytest.approx(0.8)


def test_window_summary(predictions):
    summary = summarize_birdnet_by_window(predictions)

    assert not summary.empty
    assert set(summary["window_start_seconds"]) == {0.0, 3.0}


def test_fusion_output_is_finite(predictions):
    vggish = np.ones((7, 128), dtype=np.float32)
    efficientnet = np.ones((1, 1280), dtype=np.float32)

    fused = fuse_modalities_by_window(
        predictions,
        vggish,
        efficientnet,
    )

    assert len(fused) == 3
    assert "vggish_000" in fused.columns
    assert "efficientnet_0000" in fused.columns
    assert np.isfinite(
        fused.select_dtypes(include="number").to_numpy()
    ).all()


def test_fusion_rejects_invalid_embeddings(predictions):
    vggish = np.full((7, 128), np.nan, dtype=np.float32)
    efficientnet = np.ones((1, 1280), dtype=np.float32)

    with pytest.raises(ValueError, match="NaN or infinity"):
        fuse_modalities_by_window(
            predictions,
            vggish,
            efficientnet,
        )


def test_invalid_confidence_is_rejected(predictions):
    predictions.loc[0, "confidence"] = 1.5

    with pytest.raises(ValueError, match="between 0 and 1"):
        summarize_birdnet(predictions)
from pathlib import Path
import pytest
import pandas as pd
from backend.orchestrator import BioacousticOrchestrator


def test_orchestrator_end_to_end_analysis():
    audio_path = Path("data/recordings/house_sparrow.wav")
    assert audio_path.is_file()

    orchestrator = BioacousticOrchestrator()
    result = orchestrator.analyze(str(audio_path), force_vggish=True)

    assert result["audio_file"] == str(audio_path.resolve())
    assert result["snr_db"] > 0
    assert result["route"] in ("birdnet", "vggish", "both")

    # Modalities
    assert result["birdnet"] is not None
    assert result["birdnet"]["prediction_count"] > 0

    assert result["vggish"] is not None
    assert result["vggish"]["shape"][1] == 128

    assert result["efficientnet"] is not None
    assert result["efficientnet"]["shape"][1] == 1280

    # Fusion
    assert result["fusion"] is not None
    assert result["fusion"]["status"] == "completed"
    assert result["fusion"]["window_count"] > 0
    assert result["fusion"]["trained_classifier"] is True

    preds_path = result["fusion"]["predictions_csv_path"]
    assert preds_path is not None
    assert Path(preds_path).is_file()

    preds_df = pd.read_csv(preds_path)
    assert len(preds_df) == result["fusion"]["window_count"]
    assert "fusion_predicted_species" in preds_df.columns
    assert "fusion_confidence" in preds_df.columns

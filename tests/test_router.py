from pathlib import Path
import pytest
from backend.snr_router import choose_path, estimate_snr, route_audio, SNRResult


def test_choose_path_thresholds():
    assert choose_path(20.0) == "birdnet"
    assert choose_path(15.0) == "birdnet"
    assert choose_path(14.9) == "vggish"
    assert choose_path(5.0) == "vggish"
    assert choose_path(4.9) == "both"
    assert choose_path(-2.0) == "both"


def test_estimate_snr_sample():
    audio_path = "data/recordings/house_sparrow.wav"
    assert Path(audio_path).is_file()
    snr = estimate_snr(audio_path)
    assert isinstance(snr, float)
    assert snr > 0.0


def test_route_audio_returns_valid_result():
    audio_path = "data/recordings/house_sparrow.wav"
    result = route_audio(audio_path)
    assert isinstance(result, SNRResult)
    assert result.route in ("birdnet", "vggish", "both")
    assert isinstance(result.snr_db, float)
    assert len(result.reason) > 0


def test_route_audio_missing_file():
    with pytest.raises(FileNotFoundError):
        route_audio("non_existent_file.wav")

from backend.snr_router import choose_path, route_audio, SNRResult


def test_choose_path():
    assert choose_path(18.5) == "birdnet"
    assert choose_path(10.0) == "vggish"
    assert choose_path(4.2) == "both"
    assert choose_path(5.0) == "vggish"
    assert choose_path(15.0) == "birdnet"


def test_route_audio_sample():
    result = route_audio("data/recordings/house_sparrow.wav")
    assert isinstance(result, SNRResult)
    assert result.snr_db > 0
    assert result.route in ("birdnet", "vggish", "both")
    assert len(result.reason) > 0


if __name__ == "__main__":
    print("Clean recording:", choose_path(18.5))
    print("Noisy recording:", choose_path(4.2))

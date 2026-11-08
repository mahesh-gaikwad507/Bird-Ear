
import soundfile as sf
import sounddevice as sd

AUDIO_PATH = "data/recordings/house_sparrow.wav"
WINDOW_SECONDS = 3

audio, sample_rate = sf.read(AUDIO_PATH, dtype="float32")

if audio.ndim > 1:
    audio = audio.mean(axis=1)

duration = len(audio) / sample_rate
total_windows = int(duration // WINDOW_SECONDS) + (
    1 if duration % WINDOW_SECONDS else 0
)

print(f"Duration: {duration:.2f} seconds")
print(f"Sample rate: {sample_rate} Hz")
print(f"Windows to review: {total_windows}")

while True:
    raw = input(
        f"\nEnter window number (0-{total_windows - 1}), "
        "'n' for next, or 'q' to quit: "
    ).strip().lower()

    if raw == "q":
        break

    if raw == "n":
        print("Enter the next window number manually.")
        continue

    try:
        window_number = int(raw)
    except ValueError:
        print("Please enter a valid number.")
        continue

    if not 0 <= window_number < total_windows:
        print("Window number is out of range.")
        continue

    start = window_number * WINDOW_SECONDS
    end = min(start + WINDOW_SECONDS, duration)

    segment = audio[
        int(start * sample_rate):int(end * sample_rate)
    ]

    print(f"Playing {start:.1f}-{end:.1f} seconds")
    sd.play(segment, sample_rate)
    sd.wait()


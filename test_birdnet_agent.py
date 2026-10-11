from pathlib import Path
from backend.agents.birdnet_agent import BirdNETAgent


def main():
    audio_path = Path("data/recordings/house_sparrow.wav")

    print("Initializing BirdNET agent...")
    agent = BirdNETAgent()

    print("Analyzing recording...")
    results = agent.analyze(audio_path)

    print("\nFirst 10 prediction rows:")
    print(results.head(10).to_string(index=False))

    output_path = Path("data/recordings/birdnet_agent_predictions.csv")
    results.to_csv(output_path, index=False)

    print(f"\nSaved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
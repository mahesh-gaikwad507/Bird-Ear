import pandas as pd
from pathlib import Path

input_path = Path("data/labels/auto_window_label_queue.csv")
output_path = Path("data/labels/prioritized_review_queue.csv")

if not input_path.exists():
    raise FileNotFoundError(f"File not found: {input_path}")

df = pd.read_csv(input_path)

required = [
    "window_id",
    "window_start_seconds",
    "window_end_seconds",
    "birdnet_suggested_species",
    "birdnet_confidence",
    "listener_observation",
    "verified_species",
    "review_status",
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["birdnet_confidence"] = pd.to_numeric(
    df["birdnet_confidence"], errors="coerce"
)

df = df.sort_values("window_start_seconds").reset_index(drop=True)

# Compare each window's prediction with the preceding window.
previous_species = df["birdnet_suggested_species"].shift(1)

df["species_changed"] = (
    df["birdnet_suggested_species"].notna()
    & previous_species.notna()
    & df["birdnet_suggested_species"].ne(previous_species)
)

# Flag windows with listener observations.
df["has_listener_observation"] = (
    df["listener_observation"].fillna("").str.strip().ne("")
)

# Assign a confidence category; confidence is a model score,
# not a calibrated probability that the label is correct.
df["confidence_band"] = pd.cut(
    df["birdnet_confidence"],
    bins=[-float("inf"), 0.40, 0.70, float("inf")],
    labels=["low", "medium", "high"],
    right=False,
).astype("string")

# Higher priority score means review sooner.
df["review_priority_score"] = (
    (df["birdnet_confidence"] < 0.40).fillna(False).astype(int) * 3
    + df["species_changed"].astype(int) * 2
    + df["has_listener_observation"].astype(int) * 2
)

# Already verified windows are kept, but excluded from the review queue.
df["review_priority"] = df["review_status"].eq("verified").map(
    {True: "already_verified", False: "review"}
)

df["review_priority_score"] = df["review_priority_score"].where(
    df["review_priority"] == "review", 0
)

# Sort unverified, higher-priority windows first.
df = df.sort_values(
    ["review_priority", "review_priority_score", "window_start_seconds"],
    ascending=[True, False, True],
    kind="stable",
).reset_index(drop=True)

df.to_csv(output_path, index=False)

print(f"Created: {output_path}")
print(f"Total windows: {len(df)}")
print(f"Already verified: {(df['review_priority'] == 'already_verified').sum()}")
print(f"Need review: {(df['review_priority'] == 'review').sum()}")
print(f"Low confidence (< 0.40): {(df['birdnet_confidence'] < 0.40).sum()}")
print(f"Species changes: {df['species_changed'].sum()}")
print(f"Windows with listener notes: {df['has_listener_observation'].sum()}")

print("\nTop 15 windows to review:")
columns = [
    "window_id",
    "window_start_seconds",
    "window_end_seconds",
    "birdnet_suggested_species",
    "birdnet_confidence",
    "confidence_band",
    "species_changed",
    "listener_observation",
    "review_priority_score",
]
print(df.loc[df["review_priority"] == "review", columns].head(15).to_string(index=False))

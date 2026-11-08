import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from pydantic import BaseModel
import pandas as pd

from backend.orchestrator import BioacousticOrchestrator
from backend.snr_router import route_audio
from backend.classifier import TrimodalClassifier, bootstrap_training


app = FastAPI(
    title="Bioacoustic Trimodal Audio Intelligence API",
    description="REST API for adaptive audio routing, BirdNET + VGGish + EfficientNet fusion, and active learning.",
    version="1.0.0",
)


class RouteRequest(BaseModel):
    audio_path: str


class LabelSubmission(BaseModel):
    recording_path: str
    start_time_seconds: float
    end_time_seconds: float
    verified_species: str
    review_status: str = "verified"
    review_notes: str = ""


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "bioacoustic-trimodal", "version": "1.0.0"}


@app.post("/api/route")
def get_audio_route(req: RouteRequest):
    path = Path(req.audio_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {path}")
    result = route_audio(str(path))
    return {
        "audio_path": str(path.resolve()),
        "snr_db": round(result.snr_db, 2),
        "route": result.route,
        "reason": result.reason,
    }


@app.post("/api/analyze")
def analyze_recording(
    audio_path: str = Query(..., description="Path to WAV audio recording"),
    force_vggish: bool = Query(True, description="Force VGGish feature extraction"),
):
    path = Path(audio_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {path}")

    orchestrator = BioacousticOrchestrator()
    report = orchestrator.analyze(str(path), force_vggish=force_vggish)
    return report


@app.get("/api/queue")
def get_review_queue(limit: int = 50):
    queue_path = Path("data/labels/prioritized_review_queue.csv")
    if not queue_path.is_file():
        raise HTTPException(status_code=404, detail="Review queue not generated yet.")
    df = pd.read_csv(queue_path).head(limit)
    return df.to_dict(orient="records")


@app.post("/api/label")
def submit_label(submission: LabelSubmission):
    v_path = Path("data/labels/verified_window_labels.csv")
    df = pd.read_csv(v_path) if v_path.is_file() else pd.DataFrame(
        columns=["recording_path", "start_time_seconds", "end_time_seconds", "verified_species", "review_status", "review_notes"]
    )
    new_row = submission.dict()
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True).drop_duplicates(
        subset=["start_time_seconds", "end_time_seconds"], keep="last"
    )
    df.to_csv(v_path, index=False)
    return {"status": "success", "message": "Verified label recorded.", "data": new_row}


@app.post("/api/retrain")
def retrain_classifier():
    classifier = bootstrap_training()
    return {"status": "success", "metrics": classifier.metrics_}


@app.get("/api/benchmark")
def get_benchmark():
    rep_path = Path("data/results/evaluation_benchmark_report.json")
    if not rep_path.is_file():
        raise HTTPException(status_code=404, detail="Benchmark report not found. Run evaluate.py first.")
    with open(rep_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="127.0.0.1", port=8000, reload=True)

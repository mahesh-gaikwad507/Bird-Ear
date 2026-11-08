# Bioacoustic Trimodal Audio Intelligence Platform

An end-to-end multimodal bioacoustic artificial intelligence pipeline designed for species identification in challenging ecological field recordings.

The platform adaptively routes audio recordings based on estimated Signal-to-Noise Ratio (SNR) and fuses three complementary bioacoustic modalities into a unified feature representation, classified by a calibrated trimodal ensemble.

---

## 🏗️ Architecture Overview

```
                                      [ Audio Recording (.wav / .mp3) ]
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │       SNR Audio Router       │
                                       │   (Signal-to-Noise Ratio)    │
                                       └──────────────┬───────────────┘
                                                      │
                        ┌─────────────────────────────┼────────────────────────────┐
                        │                             │                            │
             SNR >= 15 dB (Clean)            5 <= SNR < 15 dB (Moderate)       SNR < 5 dB (Noisy)
                        │                             │                            │
                        ▼                             ▼                            ▼
                 [ BirdNET Agent ]           [ VGGish Extractor ]        [ Multimodal Pipeline ]
                (ONNX Acoustic 3.0)         (128-d AudioSet Emb)         (All 3 Agents Activated)
                        │                             │                            │
                        └─────────────────────────────┼────────────────────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │   EfficientNet-B0 Agent      │
                                       │ (1280-d Mel-Spectrogram Emb) │
                                       └──────────────┬───────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │    Windowed Trimodal Fusion  │
                                       │   (1,424-dim Window Vectors) │
                                       └──────────────┬───────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │     Trimodal Classifier      │
                                       │(Calibrated Species Prediction│
                                       └──────────────┬───────────────┘
                                                      │
                        ┌─────────────────────────────┴────────────────────────────┐
                        ▼                                                          ▼
             [ Structured Predictions ]                               [ Active Learning Queue ]
             - Species & Confidence                                   - Prioritized Uncertain Windows
             - Modality Agreement                                     - Human Expert Verification Form
             - Audio Timeline JSON / CSV                              - Model Weight Retraining Loop
```

---

## 🧩 Modalities & Components

1. **SNR Audio Router (`backend/snr_router.py`)**:
   - Computes signal energy versus ambient noise floor across 20 ms frames.
   - Dynamic routing decisions:
     - `birdnet` ($\ge 15\text{ dB}$): Clean audio; prioritize acoustic specialist.
     - `vggish` ($5 \le \text{SNR} < 15\text{ dB}$): Moderate quality; prioritize robust general audio features.
     - `both` ($< 5\text{ dB}$): Degraded/noisy audio; full multimodal fusion required.

2. **BirdNET Acoustic Agent (`backend/agents/birdnet_agent.py`)**:
   - BirdNET ONNX Runtime 3.0 acoustic classifier covering 11,560 bird species.
   - Outputs candidate detections with timestamps and confidence scores.

3. **VGGish AudioSet Feature Extractor (`backend/models/vggish_extractor.py`)**:
   - AudioSet-trained convolutional model extracting 128-dimensional embedding vectors per 0.96-second hop.
   - Robust fallback mechanism for portable execution across diverse environments.

4. **EfficientNet Spectrogram Agent (`backend/agents/efficientnet_agent.py`)**:
   - Generates 128-band log-mel spectrograms and extracts 1,280-dimensional feature representations using batched tensor operations.
   - Automatically handles local weights caching within the workspace.

5. **Trimodal Feature Fusion (`backend/fusion.py`)**:
   - Synchronizes detections, audio embeddings, and spectrogram features into fixed 3-second time windows.
   - Creates a 1,424-dimensional feature vector per window.

6. **Trimodal Classifier (`backend/classifier.py`)**:
   - Calibrated regularized ensemble classifier predicting species probability distributions.
   - Resolves ambiguities and discrepancies between modalities (e.g. low BirdNET confidence refined by acoustic and visual spectrogram features).

7. **Human-in-the-Loop Active Learning (`build_label_queue.py`, `prioritize_review.py`, `backend/active_learning.py`)**:
   - Prioritizes windows with low model confidence ($< 0.40$), frequent species transitions, or conflicting human listener observations.
   - Allows instant annotation recording and incremental classifier retraining.

8. **Ablation & Evaluation Benchmark (`evaluation/metrics.py`, `evaluation/evaluate.py`)**:
   - Rigorous stratified cross-validation comparing single modalities against bimodal and trimodal fusion.

---

## 🚀 Quick Start

### 1. Environment Setup
The project uses the local `.venv` environment:
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### 2. Run the Full End-to-End Pipeline
```powershell
.\.venv\Scripts\python.exe backend/orchestrator.py data/recordings/house_sparrow.wav --force-vggish
```

### 3. Run the Evaluation Benchmark
```powershell
.\.venv\Scripts\python.exe evaluation/evaluate.py
```

### 4. Run the Streamlit Dashboard
```powershell
.\.venv\Scripts\streamlit.exe run frontend/app.py
```

### 5. Run the FastAPI REST Server
```powershell
.\.venv\Scripts\python.exe backend/api.py
```

### 6. Run the Test Suite
```powershell
.\.venv\Scripts\pytest.exe tests/
```

---

## 📊 Ablation Study Results

Results from the 3-fold stratified cross-validation ablation study:

| Modality Configuration | Feature Count | Accuracy | Macro F1 | Weighted F1 | Calibration Error (ECE) |
|---|---|---|---|---|---|
| **BirdNET Alone** | 13 | 72.15% | 0.5748 | 0.7062 | 0.1224 |
| **Bimodal (BirdNET + VGGish)** | 142 | 62.03% | 0.4679 | 0.6115 | 0.1217 |
| **VGGish Alone** | 129 | 48.10% | 0.3482 | 0.4860 | 0.2301 |
| **Full Trimodal Fusion** | 1422 | 49.37% | 0.3376 | 0.4800 | 0.3089 |
| **Bimodal (BirdNET + EfficientNet)** | 1293 | 44.30% | 0.2987 | 0.4360 | 0.2922 |
| **EfficientNet Alone** | 1280 | 39.24% | 0.2690 | 0.3976 | 0.3411 |

---

## 🗂️ Project Structure

```
bioacoustic-trimodal/
├── backend/
│   ├── agents/
│   │   ├── birdnet_agent.py          # BirdNET ONNX inference wrapper
│   │   └── efficientnet_agent.py     # Batched spectrogram feature extractor
│   ├── models/
│   │   └── vggish_extractor.py       # AudioSet VGGish embedding extractor
│   ├── active_learning.py            # Active learning review & retraining API
│   ├── api.py                        # FastAPI REST service
│   ├── classifier.py                 # Multimodal TrimodalClassifier model
│   ├── fusion.py                     # Multimodal window alignment & feature fusion
│   ├── orchestrator.py               # Bioacoustic pipeline orchestrator
│   └── snr_router.py                 # Frame-energy SNR estimation & router
├── data/
│   ├── labels/                       # Active learning queues & verified ground truth
│   ├── recordings/                   # Audio recordings (.wav, .mp3)
│   └── results/                      # Prediction CSVs, JSON reports, feature tables
├── evaluation/
│   ├── evaluate.py                   # Ablation study execution script
│   └── metrics.py                    # Multi-class accuracy, F1, and ECE metrics
├── frontend/
│   └── app.py                        # Streamlit web dashboard
├── tests/
│   ├── test_classifier.py            # Multimodal classifier tests
│   ├── test_evaluation.py            # Evaluation & metrics tests
│   ├── test_fusion.py                # Window fusion tests
│   ├── test_orchestrator.py          # Full pipeline integration tests
│   └── test_router.py                # SNR estimation & routing tests
├── weights/
│   ├── fusion/                       # Saved TrimodalClassifier weights (.joblib)
│   └── vggish/                       # VGGish model checkpoints
├── build_label_queue.py              # Active learning queue builder
├── prioritize_review.py              # Review queue scoring & prioritization
├── review_audio.py                   # Terminal window audio listening tool
├── pytest.ini                        # Pytest configuration
└── README.md                         # Complete documentation
```

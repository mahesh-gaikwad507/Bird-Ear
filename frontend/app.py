import io
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from backend.orchestrator import BioacousticOrchestrator
from backend.snr_router import estimate_snr, route_audio
from backend.classifier import TrimodalClassifier, DEFAULT_MODEL_PATH, bootstrap_training


st.set_page_config(
    page_title="Bioacoustic Trimodal Intelligence",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F3F4F6; border-radius: 8px; padding: 16px; border-left: 4px solid #3B82F6; }
    .badge-birdnet { background-color: #D1FAE5; color: #065F46; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
    .badge-vggish { background-color: #FEF3C7; color: #92400E; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
    .badge-both { background-color: #FEE2E2; color: #991B1B; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">🦜 Bioacoustic Trimodal Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Adaptive Bioacoustic Routing • BirdNET + VGGish + EfficientNet Fusion • Active Learning Queue</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration & Input")

sample_recordings = list(Path("data/recordings").glob("*.wav"))
sample_names = [f.name for f in sample_recordings]
default_idx = sample_names.index("house_sparrow.wav") if "house_sparrow.wav" in sample_names else 0

input_mode = st.sidebar.radio("Audio Source", ["Preset Sample", "Upload WAV/MP3"], index=0)

if input_mode == "Preset Sample" and sample_names:
    selected_sample = st.sidebar.selectbox("Select Recording", sample_names, index=default_idx)
    audio_path = Path("data/recordings") / selected_sample
else:
    uploaded_file = st.sidebar.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg", "flac"])
    if uploaded_file is not None:
        save_path = Path("data/recordings") / uploaded_file.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        audio_path = save_path
    else:
        audio_path = Path("data/recordings/house_sparrow.wav")

force_vggish = st.sidebar.checkbox("Force VGGish Feature Extraction", value=True, help="Extract AudioSet VGGish features regardless of SNR route")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Model Components")
st.sidebar.markdown("- **BirdNET**: Species Acoustic ONNX Model")
st.sidebar.markdown("- **VGGish**: 128-d AudioSet Embeddings")
st.sidebar.markdown("- **EfficientNet-B0**: 1280-d Mel-Spectrogram Features")
st.sidebar.markdown("- **Trimodal Classifier**: Fused 1424-d Calibrated Ensemble")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎙️ Audio & SNR Routing",
    "📊 Trimodal Detections",
    "🎧 Active Learning Review",
    "📈 Multimodal Benchmark",
])

# ----------------------------------------------------
# TAB 1: AUDIO & SNR ROUTING
# ----------------------------------------------------
with tab1:
    st.subheader("Audio Inspection & Dynamic Quality Router")

    if audio_path.is_file():
        audio_bytes = open(audio_path, "rb").read()
        st.audio(audio_bytes)

        # SNR Estimation
        routing = route_audio(str(audio_path))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Estimated Signal-to-Noise Ratio", f"{routing.snr_db:.2f} dB")
        with c2:
            route_label = routing.route.upper()
            badge_class = f"badge-{routing.route.lower()}"
            st.markdown(f"**Assigned Pipeline Route:**")
            st.markdown(f'<span class="{badge_class}">{route_label}</span>', unsafe_allow_html=True)
        with c3:
            st.markdown(f"**Routing Policy Rationale:**")
            st.info(routing.reason)

        st.markdown("---")
        run_btn = st.button("🚀 Run Bioacoustic Pipeline", type="primary")

        if run_btn:
            with st.spinner("Executing dynamic routing, feature extraction, and trimodal fusion..."):
                orchestrator = BioacousticOrchestrator()
                report = orchestrator.analyze(str(audio_path), force_vggish=force_vggish)
                st.session_state["pipeline_report"] = report
                st.success("✅ Multimodal Analysis Completed Successfully!")

        if "pipeline_report" in st.session_state:
            report = st.session_state["pipeline_report"]
            st.json(report, expanded=False)
    else:
        st.warning("Selected audio file does not exist.")

# ----------------------------------------------------
# TAB 2: TRIMODAL DETECTIONS
# ----------------------------------------------------
with tab2:
    st.subheader("Window-Level Trimodal Predictions")

    fused_csv = Path(f"data/results/{audio_path.stem}_trimodal_predictions.csv")
    if not fused_csv.is_file():
        # Fallback to test fused file if available
        fused_csv = Path("data/results/house_sparrow_trimodal_predictions.csv")

    if fused_csv.is_file():
        preds_df = pd.read_csv(fused_csv)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Analysis Windows", len(preds_df))
        with m2:
            top_species = preds_df["fusion_predicted_species"].mode()[0]
            st.metric("Dominant Species", top_species)
        with m3:
            avg_conf = preds_df["fusion_confidence"].mean()
            st.metric("Mean Fusion Confidence", f"{avg_conf:.1%}")
        with m4:
            agreed = preds_df["modalities_agree"].sum()
            st.metric("BirdNET Agreement", f"{agreed}/{len(preds_df)} ({agreed/len(preds_df):.1%})")

        st.markdown("#### Detection Timeline")
        timeline_df = preds_df[["window_start_seconds", "window_end_seconds", "fusion_predicted_species", "fusion_confidence", "birdnet_suggested_species", "birdnet_confidence", "modalities_agree", "confidence_gain"]]

        # Color highlight for disagreement
        st.dataframe(
            timeline_df.style.apply(
                lambda row: ["background-color: #FEF3C7" if not row["modalities_agree"] else "" for _ in row],
                axis=1
            ),
            use_container_width=True,
        )

        st.download_button(
            "📥 Download Predictions CSV",
            preds_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{audio_path.stem}_trimodal_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("Run the pipeline on Tab 1 to view detection results.")

# ----------------------------------------------------
# TAB 3: ACTIVE LEARNING & HUMAN REVIEW
# ----------------------------------------------------
with tab3:
    st.subheader("Human-in-the-Loop Active Learning Review Queue")
    st.markdown("Prioritize expert acoustic verification for uncertain windows (low BirdNET confidence, species changes, or human observation conflicts).")

    queue_path = Path("data/labels/prioritized_review_queue.csv")
    if queue_path.is_file():
        queue_df = pd.read_csv(queue_path)

        st.dataframe(
            queue_df[["window_id", "window_start_seconds", "window_end_seconds", "birdnet_suggested_species", "birdnet_confidence", "confidence_band", "species_changed", "listener_observation", "review_priority_score"]].head(25),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("🎧 Window Audio Verification Form")

        c1, c2 = st.columns([1, 2])
        with c1:
            window_opts = queue_df["window_id"].tolist()
            selected_wid = st.selectbox("Select Window ID to Review", window_opts)
            win_row = queue_df[queue_df["window_id"] == selected_wid].iloc[0]

            w_start = float(win_row["window_start_seconds"])
            w_end = float(win_row["window_end_seconds"])

            # Play sliced window audio
            if audio_path.is_file():
                try:
                    data, sr = sf.read(str(audio_path), dtype="float32")
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    s_idx = int(w_start * sr)
                    e_idx = int(w_end * sr)
                    segment = data[s_idx:e_idx]

                    buf = io.BytesIO()
                    sf.write(buf, segment, sr, format="WAV")
                    st.audio(buf.getvalue(), format="audio/wav")
                    st.caption(f"Playing window {selected_wid}: {w_start:.1f}s - {w_end:.1f}s")
                except Exception as e:
                    st.warning(f"Audio playback error: {e}")

        with c2:
            st.markdown(f"**Current Suggestion:** {win_row.get('birdnet_suggested_species', 'N/A')}")
            st.markdown(f"**BirdNET Confidence:** {win_row.get('birdnet_confidence', 0.0):.2f}")
            st.markdown(f"**Listener Notes:** {win_row.get('listener_observation', 'None')}")

            verified_species = st.text_input("Verified Species Name", value=str(win_row.get("verified_species", "") or win_row.get("birdnet_suggested_species", "")))
            review_notes = st.text_input("Reviewer Notes", value="Verified via interactive review")

            if st.button("💾 Save Verified Label", type="primary"):
                # Update verified labels
                v_path = Path("data/labels/verified_window_labels.csv")
                v_df = pd.read_csv(v_path) if v_path.is_file() else pd.DataFrame(columns=["recording_path", "start_time_seconds", "end_time_seconds", "verified_species", "review_status", "review_notes"])

                new_row = {
                    "recording_path": str(audio_path),
                    "start_time_seconds": w_start,
                    "end_time_seconds": w_end,
                    "verified_species": verified_species,
                    "review_status": "verified",
                    "review_notes": review_notes,
                }
                v_df = pd.concat([v_df, pd.DataFrame([new_row])], ignore_index=True).drop_duplicates(subset=["start_time_seconds", "end_time_seconds"], keep="last")
                v_df.to_csv(v_path, index=False)
                st.success(f"Saved verified label for window {selected_wid} ({w_start}s - {w_end}s) -> {verified_species}")

        st.markdown("---")
        if st.button("🔄 Retrain Trimodal Classifier with Updated Labels"):
            with st.spinner("Retraining multimodal classifier with latest active learning dataset..."):
                cls = bootstrap_training()
                st.success("✅ Trimodal Classifier Retrained and Saved!")
    else:
        st.info("Prioritized review queue not found. Run build_label_queue.py first.")

# ----------------------------------------------------
# TAB 4: BENCHMARK & MULTIMODAL ABLATION
# ----------------------------------------------------
with tab4:
    st.subheader("Multimodal Ablation Study & Performance Benchmarking")
    st.markdown("Comparative performance analysis demonstrating the quantitative lift from trimodal fusion versus single modalities.")

    ablation_csv = Path("data/results/ablation_study_summary.csv")
    if ablation_csv.is_file():
        summary_df = pd.read_csv(ablation_csv)
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("#### Weighted F1-Score by Modality")
        st.bar_chart(summary_df.set_index("Modality Configuration")["Weighted F1"])

        st.markdown("#### Accuracy by Modality")
        st.bar_chart(summary_df.set_index("Modality Configuration")["Accuracy"])

        report_json = Path("data/results/evaluation_benchmark_report.json")
        if report_json.is_file():
            with open(report_json, "r", encoding="utf-8") as f:
                rep_data = json.load(f)
            st.download_button(
                "📥 Download Full Evaluation Report (JSON)",
                json.dumps(rep_data, indent=2),
                file_name="evaluation_benchmark_report.json",
                mime="application/json",
            )
    else:
        st.info("Run evaluation/evaluate.py to generate benchmark results.")

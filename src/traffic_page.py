"""Traffic showcase UI using the shared processing engine and actual saved artifacts."""

from pathlib import Path
import json
import uuid
import pandas as pd
import streamlit as st
from .traffic import VEHICLE_CLASSES
from .video_processor import processVideo


def renderTrafficPage(root):
    st.subheader("Traffic Analytics")
    st.caption(
        "Vehicle presence, classes and directional crossings from video. A showcase use case within Vision Intelligence."
    )
    with st.expander("Source and analysis settings"):
        left, right = st.columns(2)
        with left:
            source = st.radio(
                "Traffic input",
                ["Traffic showcase", "Upload traffic MP4"],
                horizontal=True,
            )
            upload = (
                st.file_uploader(
                    "Clean traffic footage", type=["mp4"], key="trafficUpload"
                )
                if source == "Upload traffic MP4"
                else None
            )
            selected = st.multiselect(
                "Vehicle classes", list(VEHICLE_CLASSES), default=list(VEHICLE_CLASSES)
            )
            confidence = st.slider("Vehicle confidence", 0.1, 0.8, 0.3, 0.05)
        with right:
            orientation = st.selectbox(
                "Counting line orientation", ["horizontal", "vertical"]
            )
            position = st.slider("Counting line position", 0.1, 0.9, 0.55, 0.05)
            trails = st.checkbox("Short movement trails", True)
            st.caption(
                "Horizontal: A = top to bottom, B = bottom to top. Vertical: A = left to right, B = right to left. Each track ID is counted once per run."
            )
        st.caption(
            "Use clean footage, not a previously annotated export. For non-vehicle classes, use Deep Vision."
        )
        if st.button("Process traffic video", type="primary"):
            if not selected:
                st.warning("Select at least one vehicle class.")
            elif source == "Upload traffic MP4" and upload is None:
                st.warning("Choose an MP4 first.")
            else:
                folder = root / "outputs" / "runs" / uuid.uuid4().hex
                folder.mkdir(parents=True)
                path = root / "data" / "cars.mp4"
                if upload is not None:
                    path = folder / "input.mp4"
                    path.write_bytes(upload.getvalue())
                try:
                    progress = st.progress(0, text="Loading model")
                    processVideo(
                        path,
                        folder,
                        mode="detection",
                        confidence=confidence,
                        traffic=True,
                        line=position,
                        lineOrientation=orientation,
                        vehicleClasses=selected,
                        trails=trails,
                        progress=lambda n, total: progress.progress(
                            min(n / max(total, 1), 1),
                            text=f"Analyzing frame {n} / {total}",
                        ),
                    )
                    st.session_state.trafficRunDir = str(folder)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Traffic analysis failed: {exc}")
        if st.button("Load saved traffic showcase"):
            st.session_state.trafficRunDir = str(root / "outputs" / "traffic")
            st.rerun()

    folder = Path(st.session_state.get("trafficRunDir", root / "outputs" / "traffic"))
    if not (folder / "run.json").exists():
        st.info("Process the supplied traffic clip to create measured results.")
        return
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    summary = run["summary"]
    st.session_state.runDir = str(folder)
    df = pd.read_csv(folder / "frame_metrics.csv")
    st.caption(
        f"COMPLETED TRAFFIC RUN / {summary['total_frames']} FRAMES / {summary['source_fps']:g} SOURCE FPS / {summary['duration_seconds']:.2f} SEC"
    )
    media, metrics = st.columns([3.2, 1], gap="large")
    with media:
        st.video(str(folder / run["video"]))
        st.caption(
            "Playback uses source FPS. Metrics below summarize the completed run; the video HUD shows per-frame values."
        )
    with metrics:
        st.metric("Total crossings", summary["total_crossings"])
        a, b = st.columns(2)
        a.metric("Direction A", summary["direction_a"])
        b.metric("Direction B", summary["direction_b"])
        st.metric("Processing FPS", f"{summary['aggregate_processing_fps']:.1f}")
        st.metric("Mean latency (ms)", f"{summary['average_processing_time_ms']:.1f}")
    a, b, c, d = st.columns(4)
    a.metric("Mean active vehicles", f"{summary['mean_active_vehicles']:.1f}")
    b.metric("Peak active vehicles", summary["peak_active_vehicles"])
    c.metric("Mean inference (ms)", f"{summary['mean_inference_time_ms']:.1f}")
    d.metric("P95 latency (ms)", f"{summary['p95_processing_time_ms']:.1f}")
    with st.expander("Vehicle classes and frame telemetry", expanded=True):
        a, b = st.columns([1, 1.6])
        rows = [
            {
                "Class": name,
                "Mean visible": round(values["mean_visible"], 2),
                "Peak visible": values["peak_visible"],
            }
            for name, values in summary["vehicle_classes"].items()
        ]
        a.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        b.line_chart(
            df.set_index("timestamp_seconds")[["active_vehicles"]],
            color=["#79a9bd"],
            height=210,
        )
        a.caption("These are per-frame counts, not counts of unique physical vehicles.")
    st.caption(
        f"Saved line: {run['line_orientation']} at {run['line']:.0%}. Confidence: {run['confidence']:.2f}. No ground-truth accuracy is claimed for this clip."
    )
    with st.expander("Crossing event audit"):
        st.dataframe(
            pd.read_csv(folder / "crossing_events.csv"),
            hide_index=True,
            width="stretch",
        )
    st.caption(
        "Export Center includes the video, per-frame telemetry and timestamped crossing events. CPU throughput excludes decode, encoding and disk I/O; first inference remains included."
    )

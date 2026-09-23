"""Vision Intelligence - local, artifact-backed Streamlit dashboard."""

from pathlib import Path
import json
import uuid
import cv2
import pandas as pd
import streamlit as st
from src.preprocessing import VisionConfig
from src.video_processor import processVideo, FrameProcessor

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Vision Intelligence", page_icon="V", layout="wide")
st.markdown(
    """<style>
.block-container {max-width:1400px;padding-top:4.5rem;}
h1 {letter-spacing:-1.8px;font-weight:750;}
[data-testid="stMetric"] {background:#141f30;border:1px solid #27364a;border-radius:10px;padding:16px;}
[data-testid="stMetricLabel"] {color:#a5b7cb;}
[data-testid="stMetricValue"] {font-size:1.65rem;}
.eyebrow {color:#36c9b0;font-size:12px;letter-spacing:2px;font-weight:700;}
.intro {color:#a5b7cb;font-size:17px;margin-bottom:22px;}
.cap {border-top:2px solid #36c9b0;padding:12px 0;color:#bacadc;}
</style>""",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="eyebrow">COMPUTER VISION / EXPERIMENT WORKSPACE</div>',
    unsafe_allow_html=True,
)
st.title("Vision Intelligence")
st.markdown(
    '<div class="intro">Real-Time Detection  |  Segmentation  |  Tracking  |  Analytics</div>',
    unsafe_allow_html=True,
)
with st.sidebar:
    st.markdown("### Vision workspace")
    page = st.radio(
        "Navigate",
        [
            "Traffic Analytics",
            "Live Vision Pipeline",
            "Frame Analysis",
            "Performance Dashboard",
            "Deep Vision",
            "Export Center",
            "About Project",
        ],
    )
    st.divider()
    st.caption("PROGREE / TASK 4")
    st.caption(
        "Classical CV is the required pipeline. Pretrained YOLO is an additional feature."
    )
    st.caption("Local processing  |  CPU inference  |  measured telemetry")

runDir = Path(st.session_state.get("runDir", ROOT / "outputs"))
manifest = (
    json.loads((runDir / "run.json").read_text())
    if (runDir / "run.json").exists()
    else None
)


def showMetrics(run):
    summary = run["summary"]
    columns = st.columns(5)
    for col, (title, value) in zip(
        columns,
        [
            ("Frames", summary["total_frames"]),
            ("Latency (ms)", f"{summary['average_processing_time_ms']:.2f}"),
            ("Processing FPS", f"{summary['aggregate_processing_fps']:.1f}"),
            (
                "Peak vehicles" if run.get("traffic") else "Peak objects",
                summary["maximum_objects_detected"],
            ),
            ("Crossings", summary["total_crossings"])
            if run.get("traffic")
            else ("Unique IDs", summary["unique_tracks"]),
        ],
    ):
        col.metric(title, value)


def uploadPath(upload, folder):
    # Ignore the supplied filename; isolate each session run on disk.
    path = folder / "input.mp4"
    path.write_bytes(upload.getvalue())
    return path


if page == "Traffic Analytics":
    from src.traffic_page import renderTrafficPage

    renderTrafficPage(ROOT)
elif page == "Live Vision Pipeline":
    for col, title, subtitle in zip(
        st.columns(4),
        [
            "Classical Vision",
            "Deep Detection",
            "Multi-Object Tracking",
            "Frame Analytics",
        ],
        [
            "Gaussian + adaptive masks",
            "Pretrained YOLOv8",
            "Persistent IDs + trails",
            "Latency, counts and ROI",
        ],
    ):
        col.markdown(
            f'<div class="cap"><strong>{title}</strong><br>{subtitle}</div>',
            unsafe_allow_html=True,
        )
    st.subheader("Contour segmentation lab")
    left, right = st.columns([1, 2.25], gap="large")
    with left:
        source = st.radio("Video source", ["Built-in demo", "Upload MP4"])
        upload = (
            st.file_uploader("Choose MP4", type=["mp4"])
            if source == "Upload MP4"
            else None
        )
        with st.expander("Preprocessing controls", expanded=True):
            gaussian = st.select_slider("Gaussian kernel", [3, 5, 7], value=5)
            method = st.selectbox("Adaptive method", ["Gaussian", "Mean"])
            block = st.slider("Threshold block size", 3, 101, 31, 2)
            constant = st.slider("Threshold C", -5, 20, 7)
            morph = st.select_slider(
                "Morphology kernel (0 disables)", [0, 3, 5, 7], value=3
            )
            minimum = st.number_input("Minimum contour area", 0, 100000, 250)
            maximum = st.number_input("Maximum contour area", 1, 500000, 20000)
        with st.expander("Tracking and scene analytics"):
            distance = st.slider("Association distance (px)", 10, 150, 55)
            disappeared = st.slider("Retain missing tracks (frames)", 0, 30, 10)
            trails = st.checkbox("Movement trails", True)
            useRoi = st.checkbox("Count inside ROI", True)
            horizontal = st.slider("ROI horizontal bounds", 0.0, 1.0, (0.2, 0.8), 0.05)
            vertical = st.slider("ROI vertical bounds", 0.0, 1.0, (0.1, 0.9), 0.05)
            useLine = st.checkbox("Count line crossings", True)
            line = st.slider("Vertical line position", 0.1, 0.9, 0.5, 0.05)
        if st.button("Process video", type="primary", width="stretch"):
            if source == "Upload MP4" and upload is None:
                st.warning("Choose an MP4 before processing.")
            else:
                folder = ROOT / "outputs" / "runs" / uuid.uuid4().hex
                folder.mkdir(parents=True)
                try:
                    path = (
                        uploadPath(upload, folder)
                        if upload
                        else ROOT / "data" / "sample_shapes.mp4"
                    )
                    progress = st.progress(0, text="Processing frames")
                    config = VisionConfig(
                        gaussian,
                        block,
                        constant,
                        method,
                        morph,
                        minimum,
                        maximum,
                        distance,
                        disappeared,
                    )
                    processVideo(
                        path,
                        folder,
                        config,
                        roi=(horizontal[0], vertical[0], horizontal[1], vertical[1])
                        if useRoi
                        else None,
                        line=line if useLine else None,
                        trails=trails,
                        progress=lambda current, total: progress.progress(
                            min(current / max(total, 1), 1),
                            text=f"Frame {current} / {total}",
                        ),
                    )
                    st.session_state.runDir = str(folder)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")
    with right:
        if manifest:
            st.caption(
                f"COMPLETED RUN / {manifest['summary']['mode'].upper()} / {manifest['input_name']}"
            )
            st.video(str(runDir / manifest["video"]))
            showMetrics(manifest)
            st.caption(
                "Processing FPS excludes video decoding, disk writes and encoding. Annotated playback retains the source FPS; audio is omitted."
            )
            st.info(
                "Explore saved stages in Frame Analysis, then export the annotated video and measured CSVs."
            )
        else:
            st.info("Process the built-in demo to create your first run.")
elif page == "Frame Analysis":
    st.subheader("Pipeline inspector")
    dataset = st.radio(
        "Inspect dataset", ["Selected run", "Classical benchmark"], horizontal=True
    )
    if dataset == "Classical benchmark":
        runDir = ROOT / "outputs"
        manifest = json.loads((runDir / "run.json").read_text(encoding="utf-8"))
    st.caption(
        "Saved representative frames: changing the selection reads artifacts without rerunning the video."
    )
    if manifest:
        number = st.select_slider(
            "Representative frame", options=manifest["representative_frames"]
        )
        stages = [
            "original",
            "grayscale",
            "blurred",
            "threshold",
            "mask",
            "segmented",
            "tracked",
        ]
        available = [
            name
            for name in stages
            if (runDir / "frames" / f"frame_{number:04d}_{name}.png").exists()
        ]
        for index, col in enumerate(st.columns(2)):
            for name in available[index::2]:
                label = name.title()
                if name == "segmented":
                    label = {
                        "detection": "Detections (bounding boxes)",
                        "segmentation": "Instance segmentation masks",
                    }.get(manifest["summary"]["mode"], "Contour segmentation")
                col.image(
                    str(runDir / "frames" / f"frame_{number:04d}_{name}.png"),
                    caption=label,
                    width="stretch",
                )
        df = pd.read_csv(runDir / "frame_metrics.csv")
        st.dataframe(df[df.frame_number == number], hide_index=True)
    else:
        st.info("Process a video first.")
elif page == "Performance Dashboard":
    st.subheader("Measured performance")
    if manifest:
        showMetrics(manifest)
        summary = manifest["summary"]
        st.caption(
            f"P50 {summary['median_processing_time_ms']:.2f} ms  |  P95 {summary['p95_processing_time_ms']:.2f} ms  |  Source {summary['source_fps']:.1f} FPS"
        )
        df = pd.read_csv(runDir / "frame_metrics.csv")
        if manifest.get("traffic"):
            st.caption(
                f"First-frame processing: {df.processing_time_ms.iloc[0]:.1f} ms, included in the mean and chart. Visible counts are not congestion or density estimates."
            )
        a, b = st.columns(2)
        a.markdown("**Latency / ms**")
        a.line_chart(
            df.set_index("frame_number")[["processing_time_ms"]], color=["#36c9b0"]
        )
        b.markdown("**Processing FPS**")
        b.line_chart(df.set_index("frame_number")[["fps"]], color=["#ffc373"])
        a.markdown(
            "**Active vehicles**"
            if manifest.get("traffic")
            else "**Objects and tracks**"
        )
        a.line_chart(
            df.set_index("frame_number")[
                ["active_vehicles"]
                if manifest.get("traffic")
                else ["objects_detected", "active_tracks", "roi_objects"]
            ]
        )
        b.markdown("**Class distribution over time**")
        counts = pd.DataFrame([json.loads(x) for x in df.class_counts]).fillna(0)
        counts.index = df.frame_number
        if len(counts.columns):
            b.line_chart(counts)
        st.caption(manifest["timing_scope"])
        with st.expander("Run configuration and environment"):
            st.json(manifest)
    else:
        st.info("Process a video first.")
elif page == "Deep Vision":
    st.subheader("Pretrained vision / additional feature")
    st.caption(
        "YOLOv8n detection or YOLOv8n-seg instance masks. CPU inference; first use downloads weights. COCO classes do not include geometric shapes."
    )
    modeLabel = st.radio(
        "Deep vision mode",
        ["Object detection", "Instance segmentation"],
        horizontal=True,
    )
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    tracking = st.checkbox("ByteTrack persistent IDs (video)", True)
    upload = st.file_uploader(
        "Real-world image or MP4", type=["jpg", "jpeg", "png", "mp4"]
    )
    if st.button("Run deep vision", type="primary"):
        if upload is None:
            st.warning("Upload real-world media to run pretrained vision.")
        else:
            mode = (
                "segmentation" if modeLabel == "Instance segmentation" else "detection"
            )
            folder = ROOT / "outputs" / "runs" / uuid.uuid4().hex
            folder.mkdir(parents=True)
            try:
                with st.spinner("Loading model and processing media..."):
                    if upload.name.lower().endswith(".mp4"):
                        processVideo(
                            uploadPath(upload, folder),
                            folder,
                            mode=mode,
                            confidence=confidence,
                            tracking=tracking,
                            line=0.5 if tracking else None,
                        )
                        st.session_state.runDir = str(folder)
                        st.session_state.deepVideo = str(folder)
                        st.session_state.pop("deepImage", None)
                    else:
                        import numpy as np

                        frame = cv2.imdecode(
                            np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR
                        )
                        if frame is None:
                            raise ValueError("Invalid image")
                        stages, row, _ = FrameProcessor(
                            mode=mode, confidence=confidence, tracking=False
                        ).process(frame, 1, 1)
                        path = folder / "result.png"
                        cv2.imwrite(str(path), stages["tracked"])
                        row["mode"] = mode
                        st.session_state.deepImage = (str(path), row)
                        st.session_state.pop("deepVideo", None)
            except Exception as exc:
                st.error(f"Deep vision unavailable: {exc}")
                st.info(
                    "The classical pipeline remains available. Install requirements-yolo.txt and check the initial weight download."
                )
    if "deepImage" in st.session_state:
        path, row = st.session_state.deepImage
        st.caption(f"Completed result / {row.get('mode', 'deep vision')}")
        preview, details = st.columns([1.2, 1], gap="large")
        preview.image(path, width="stretch")
        with details:
            st.metric("Detected instances", row["objects_detected"])
            st.metric("Inference latency (ms)", f"{row['inference_time_ms']:.2f}")
            st.dataframe(
                pd.DataFrame(
                    json.loads(row["class_counts"]).items(), columns=["Class", "Count"]
                ),
                hide_index=True,
            )
            st.caption(
                "Single-image inference. Tracking IDs are assigned only to video sequences."
            )
            with st.expander("Frame telemetry"):
                st.json(row)
            st.download_button(
                "Download annotated image",
                Path(path).read_bytes(),
                "annotated_image.png",
                "image/png",
            )
    if "deepVideo" in st.session_state:
        directory = Path(st.session_state.deepVideo)
        run = json.loads((directory / "run.json").read_text())
        st.video(str(directory / run["video"]))
        showMetrics(run)
        st.caption("Video artifacts are available in Frame Analysis and Export Center.")
elif page == "Export Center":
    st.subheader("Your run, ready to share")
    if manifest:
        choices = [
            runDir / manifest["video"],
            runDir / "frame_metrics.csv",
            runDir / "summary_metrics.csv",
            runDir / "detections.csv",
            runDir / "run.json",
        ]
        choices += sorted((runDir / "charts").glob("*.png"))
        if (runDir / "crossing_events.csv").exists():
            choices.append(runDir / "crossing_events.csv")
        for path in choices:
            st.download_button(
                f"Download {path.name}", path.read_bytes(), path.name, key=str(path)
            )
        st.caption(
            "Uploads and session runs are ignored by Git. They remain on this computer under outputs/runs until you remove them."
        )
    else:
        st.info("Process a video first.")
else:
    st.subheader("A transparent vision engineering project")
    st.markdown("""**Official requirements**: Gaussian filtering, adaptive threshold matrices, pixel-bound contour extraction, dynamic class counts, frame metrics and a performance whitepaper.

**Engineering additions**: from-scratch centroid tracking, bounded trails, ROI counts, one-crossing-per-ID analytics, pretrained detection and instance masks, reproducible synthetic ground truth, exported media and interactive inspection.

**Traffic showcase**: vehicle-only detection, short trails, a horizontal or vertical counting line, Direction A/B counts and timestamped crossing events. Active vehicle counts describe visible detections, not congestion or traffic density. The showcase does not optimize signals or claim production deployment.

**Interpretation matters**: contours classify geometry, YOLO detection produces boxes, and the segmentation model produces instance masks. Runtime is not detection accuracy. The synthetic benchmark uses separated shapes and does not establish robustness to real-world occlusion.

**Tracking limits**: centroid association can switch IDs when objects cross or disappear for too long. The line counter counts each track ID once per run, not every physical crossing.

Built by Fayz Liaqat for Progree Artificial Intelligence Internship, Task 4.""")
    st.link_button(
        "Ultralytics tracking documentation",
        "https://docs.ultralytics.com/modes/track/",
    )

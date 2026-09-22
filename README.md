# Vision Intelligence

**OpenCV contour segmentation, pretrained detection, instance masks and persistent tracking - with evidence for every result.**

Built by **Fayz Liaqat** for **Progree Artificial Intelligence Internship, Task 4**. The classical pipeline satisfies Gaussian filtering, adaptive thresholds, pixel-bound extraction, dynamic count overlays and frame-level evaluation. Pretrained YOLO is a separate enhancement.

![Actual Streamlit workspace](docs/demo/streamlit_dashboard.png)

[Watch the annotated 8-second demo](outputs/videos/opencv_segmented_output.mp4) · [Read the six-page whitepaper](report/Task_4_Whitepaper.pdf) · [Read the Markdown report](report/Task_4_Whitepaper.md)

## What it does

- **Classical vision:** configurable Gaussian/adaptive stages, morphology, area bounds and geometric Circle / Rectangle / Triangle / Other labels.
- **Tracking and counting:** from-scratch centroid association, 24-point trails, ROI class counts and one-crossing-per-ID counting with a deadband.
- **Deep vision:** optional YOLOv8n detection and YOLOv8n-seg instance masks; ByteTrack IDs for videos. Missing dependencies or weights leave classical processing operational.
- **Inspection:** seven saved stages, representative frames without reprocessing, actual latency/FPS/count charts and an export center.
- **Reproducibility:** seeded video, ground truth, two executed notebooks, focused tests and saved configuration/environment metadata.

## Architecture

![Architecture](docs/demo/architecture.png)

```mermaid
flowchart LR
    A[MP4 or local camera] --> B[Grayscale + Gaussian blur]
    B --> C[Adaptive threshold + morphology]
    C --> D[External contours + area bounds]
    D --> E[Geometric labels + centroid tracker]
    A --> F[Optional YOLO boxes or instance masks]
    F --> G[ByteTrack]
    E --> H[ROI + line counts + overlays]
    G --> H
    H --> I[Annotated video + CSV + dashboard]
```

The threshold mask can have hollow interiors. External contours recover enclosed boundaries; this is **contour segmentation**, not semantic segmentation. Standard YOLO detection returns boxes. Only the segmentation checkpoint returns instance masks.

![Actual pipeline stages](docs/demo/pipeline_stages.png)

## Official requirements and additions

<!-- REQUIREMENTS_START -->
| Official requirement | Implementation and evidence |
|---|---|
| Real-time CV pipeline | FrameProcessor; optional realtime_demo.py camera preview |
| Gaussian filtering | cv2.GaussianBlur; blurred PNGs and notebook 1 |
| Adaptive threshold matrices | cv2.adaptiveThreshold; Gaussian / Mean controls |
| Shape extraction by pixel bounds | External contours; min/max contour area |
| Dynamic target count overlays | Per-class counters and cv2.putText HUD |
| Frame-by-frame inference metrics | frame_metrics.csv: one measured row per frame |
| Performance whitepaper | Six-page PDF and Task_4_Whitepaper.md |
<!-- REQUIREMENTS_END -->

Additional engineering includes deterministic ground truth, geometric tracking IDs, trails, ROI and crossing counts, dashboard controls, frame inspection, H.264 export, ByteTrack and pretrained instance segmentation. No training, database, API server or cloud infrastructure was added.

## Measured results

A complete synthetic run: **240 frames · 960 x 540 · 30 source FPS · 8 seconds · seed 42**. Mild noise, three persistent shapes and a smaller circle visible on frames 46-190. Separate lanes avoid occlusion.

<!-- RESULTS_START -->
| Metric | Measured value |
|---|---|
| Processed / source frames | 240 / 240 |
| Mean / median processing latency | 19.187 / 19.003 ms |
| P95 / maximum latency | 22.749 / 24.355 ms |
| Mean per-frame FPS | 52.742 |
| Aggregate processing FPS | 52.118 |
| Mean / maximum objects | 3.6042 / 4 |
| Unique tracked IDs | 4 |
| Entered / exited (once per ID) | 2 / 2 |
<!-- RESULTS_END -->

These measurements come from [frame_metrics.csv](outputs/frame_metrics.csv) and [run.json](outputs/run.json). Latency includes processing, association, analytics and geometric overlays; it excludes source decoding, HUD text, encoding, disk I/O and model load. Aggregate processing FPS is `1000 / mean_latency_ms`; mean per-frame FPS is reported separately. Neither is a claim of end-to-end webcam throughput. Results vary with machine load.

![Measured performance](docs/demo/performance_summary.png)

**Controlled correctness:** 865 matched object-frame instances, zero unmatched predictions, zero misses, zero count MAE and zero matched ID changes. Matching is one-to-one within 20 pixels. Shape labels were correct on all matches. This easy benchmark does **not** establish real-world accuracy, mAP or robust occlusion handling. See [synthetic_evaluation.json](outputs/synthetic_evaluation.json) for definitions.

## Pretrained vision: tested, separately scoped

Both models ran on the attributed Ultralytics bus image and a 20-frame translated-photo clip. Detection returned real boxes; segmentation returned six instance masks on the still image. Both video exports were read back successfully with persistent IDs. This verifies integration, not natural-motion tracking quality. First-inference startup remains in the CSV. No speed ranking is made against the differently sized classical benchmark.

| Detection | Instance segmentation |
|---|---|
| ![Detection](docs/demo/representative_detection.png) | ![Instance masks](docs/demo/representative_segmentation.png) |

See [YOLO verification](outputs/yolo_verification.json) and [attribution](THIRD_PARTY_NOTICES.md). Model weights are downloaded on first use and excluded from Git.

## Install and run

Python **3.13** was tested on Windows. Create an isolated environment in this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS/Linux activate with `source .venv/bin/activate`. Install [FFmpeg](https://ffmpeg.org/download.html) and put `ffmpeg` on PATH for browser-compatible H.264. Without it, OpenCV exports mp4v, which some browsers cannot play; downloads and frame images remain usable.

For optional pretrained models:

```powershell
python -m pip install -r requirements-yolo.txt
```

Open **Deep Vision**, choose detection or instance segmentation, upload JPG/PNG/MP4 and run. First use needs internet to download weights. Inference is explicitly on CPU. COCO classes do not include abstract benchmark shapes; use real-world media.

## Reproduce the evidence

```powershell
python benchmark.py
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m ruff check . --select E9,F
python -m jupyter notebook notebooks
python scripts/verify_yolo.py
python scripts/build_report.py
```

Run notebook 1 top to bottom for the classical stages. Notebook 2 processes the whole sample and regenerates canonical metrics and charts; both already contain saved outputs. Rebuild the report afterward so tables match the current run. `scripts/verify_yolo.py` retrieves the attributed source image if missing, downloads weights if needed and executes both deep modes.

For a local webcam:

```powershell
python realtime_demo.py --camera 0
```

Press **q** to exit. An unavailable camera prints an actionable message and exits with code 1. Physical-camera behavior must be verified with your hardware and permissions.

## Dashboard workflow

1. In **Live Vision Pipeline**, use the demo or upload MP4, adjust parameters and select **Process video**.
2. **Frame Analysis** inspects early, middle and later saved frames without rerunning inference.
3. **Performance Dashboard** shows real telemetry and the saved configuration.
4. **Deep Vision** runs pretrained boxes or instance masks on a real-world image/video.
5. **Export Center** downloads videos, CSVs and charts.

Each interactive run uses `outputs/runs/<generated-id>/`. Uploaded names cannot choose arbitrary paths. These local runs are ignored by Git and persist until removed. Exported videos do not retain audio.

![Actual performance dashboard](docs/demo/streamlit_performance.png)

## Project structure

```text
app.py                         Streamlit workspace
benchmark.py                   Synthetic run + quality evaluation
generate_demo.py               Seeded video + ground truth
realtime_demo.py                Optional local camera
src/
  preprocessing.py             Gray, Gaussian, threshold, morphology
  segmentation.py              Contours, bounds, geometric labels
  tracking.py                  Centroid matching and scene analytics
  yolo_detector.py             Lazy optional models + ByteTrack
  video_processor.py           Shared engine, overlays, video I/O
  metrics.py                   Statistics and figures
notebooks/                     Two executed instructional notebooks
data/                          Sample MP4, truth, attributed deep sample
outputs/                       CSVs, videos, frames, charts and metadata
report/                        Six-page PDF and Markdown whitepaper
docs/demo/                     Actual app screenshots and showcase figures
scripts/                       Deep verification and report builder
tests/                         Pipeline and Streamlit AppTest coverage
```

[Exact artifact inventory](docs/ARTIFACTS.md) · [Verification record](docs/VERIFICATION.md)

## Limitations worth understanding

- Dark foreground on a bright background is the classical default. Texture, scale and overlapping contours require retuning.
- Greedy association can switch IDs. Unique IDs need not equal unique physical objects.
- A line counts **once per track ID per run**. Entered means left-to-right; exited means right-to-left.
- ROI uses centroid inclusion, not mask overlap. Full-frame detection remains active.
- Synthetic quality is not semantic segmentation accuracy or natural-scene accuracy.
- UI processing is batch-on-demand. Camera processing is a separate local script; no browser webcam streaming, cancellation or cloud deployment is claimed.
- Large uploads consume time and local storage. The app limits uploads to 100 MB.
- CPU model startup can be slow. No GPU performance was measured.

## Demo recording

See [docs/demo/README.md](docs/demo/README.md) for a short recording sequence. Screenshots come from the actual application. Nothing has been posted to LinkedIn.

## References and attribution

[OpenCV thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) · [Ultralytics tracking](https://docs.ultralytics.com/modes/track/) · [YOLOv8 models](https://docs.ultralytics.com/models/yolov8/)

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for pretrained software and sample-media provenance.

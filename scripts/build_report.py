"""Build an evidence-backed six-page whitepaper from the canonical run."""

from pathlib import Path
import html
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.pagesizes import A4
import matplotlib.pyplot as plt
from src.metrics import createCharts

ROOT = Path(__file__).resolve().parents[1]
run = json.loads((ROOT / "outputs/run.json").read_text())
s = run["summary"]
e = json.loads((ROOT / "outputs/synthetic_evaluation.json").read_text())
yolo = json.loads((ROOT / "outputs/yolo_verification.json").read_text())
for source, target in [
    ("outputs/charts/pipeline_stages.png", "pipeline_stages.png"),
    ("outputs/frames/frame_0120_tracked.png", "representative_classical.png"),
]:
    shutil.copyfile(ROOT / source, ROOT / "docs/demo" / target)

fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis("off")
boxes = [
    (0.1, 1.6, 1.6, "INPUT\nMP4 / camera"),
    (2.2, 2.6, 3, "CLASSICAL\nGray > Gaussian > adaptive\nMorphology > contours"),
    (2.2, 0.4, 3, "PRETRAINED\nYOLOv8n boxes\nYOLOv8n-seg masks"),
    (6, 1.6, 2.2, "ASSOCIATION\nCentroid / ByteTrack\nIDs + bounded trails"),
    (9, 1.6, 2.8, "OUTPUT\nROI + line counts\nVideo / CSV / inspector"),
]
for x, y, w, label in boxes:
    ax.text(
        x + w / 2,
        y + 0.4,
        label,
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=.8", fc="#141f30", ec="#36c9b0"),
        linespacing=1.5,
    )
for start, end in [
    ((1.75, 2), (2.15, 3)),
    ((1.75, 2), (2.15, 0.8)),
    ((5.3, 3), (5.95, 2)),
    ((5.3, 0.8), (5.95, 2)),
    ((8.25, 2), (8.95, 2)),
]:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color="#9dacbf", lw=2),
    )
fig.savefig(ROOT / "docs/demo/architecture.png", dpi=170)
plt.close(fig)
df = pd.read_csv(ROOT / "outputs/frame_metrics.csv")
createCharts(df, ROOT / "outputs/charts")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
for ax, col, title, color in zip(
    axes,
    ["processing_time_ms", "fps", "objects_detected"],
    ["Latency / ms", "Processing FPS", "Objects per frame"],
    ["#36c9b0", "#ffc373", "#71acf7"],
):
    ax.plot(df.frame_number, df[col], color=color, lw=1)
    ax.set(title=title, xlabel="Frame")
    ax.grid(alpha=0.15)
fig.suptitle("VISION INTELLIGENCE / MEASURED SYNTHETIC RUN", fontsize=14)
fig.savefig(ROOT / "docs/demo/performance_summary.png", dpi=170)
plt.close(fig)

summaryRows = [
    ["Metric", "Measured value"],
    ["Processed / source frames", f"{s['total_frames']} / {s['source_frame_count']}"],
    [
        "Mean / median processing latency",
        f"{s['average_processing_time_ms']:.3f} / {s['median_processing_time_ms']:.3f} ms",
    ],
    [
        "P95 / maximum latency",
        f"{s['p95_processing_time_ms']:.3f} / {s['maximum_processing_time_ms']:.3f} ms",
    ],
    ["Mean per-frame FPS", f"{s['average_fps']:.3f}"],
    ["Aggregate processing FPS", f"{s['aggregate_processing_fps']:.3f}"],
    [
        "Mean / maximum objects",
        f"{s['average_objects_per_frame']:.4f} / {s['maximum_objects_detected']}",
    ],
    ["Unique tracked IDs", str(s["unique_tracks"])],
    ["Entered / exited (once per ID)", f"{s['entered']} / {s['exited']}"],
]
mapping = [
    ["Official requirement", "Implementation and evidence"],
    [
        "Real-time CV pipeline",
        "FrameProcessor; optional realtime_demo.py camera preview",
    ],
    ["Gaussian filtering", "cv2.GaussianBlur; blurred PNGs and notebook 1"],
    ["Adaptive threshold matrices", "cv2.adaptiveThreshold; Gaussian / Mean controls"],
    ["Shape extraction by pixel bounds", "External contours; min/max contour area"],
    ["Dynamic target count overlays", "Per-class counters and cv2.putText HUD"],
    [
        "Frame-by-frame inference metrics",
        "frame_metrics.csv: one measured row per frame",
    ],
    ["Performance whitepaper", "Six-page PDF and Task_4_Whitepaper.md"],
]
pages = [
    (
        "Vision Intelligence",
        [
            (
                "subtitle",
                "A Real-Time Multi-Object Detection, Segmentation, Tracking and Performance Analytics Engine",
            ),
            (
                "p",
                "Fayz Liaqat | Progree Artificial Intelligence Internship | Task 4 | September 2026",
            ),
            ("h", "Executive summary"),
            (
                "p",
                f"This project implements the required OpenCV contour pipeline and a separate pretrained deep-vision extension. The classical path combines Gaussian filtering, adaptive thresholding, morphological cleanup, area-bounded contour extraction, geometric classification and a centroid tracker written from scratch. Streamlit exposes parameters, saved stages, telemetry and exports. The complete synthetic run processed {s['total_frames']} frames at 960 x 540 pixels. Mean processing latency was {s['average_processing_time_ms']:.3f} ms; aggregate processing throughput was {s['aggregate_processing_fps']:.2f} FPS on this machine. These are processing measurements, not a claim of end-to-end camera or deployment speed.",
            ),
            ("h", "Problem statement and scope"),
            (
                "p",
                "The official task asks for a video pipeline that extracts and tracks target contours, applies Gaussian filters and adaptive threshold matrices, overlays dynamic target counts, evaluates every frame, and documents the results. The practical engineering challenge is to connect these stages while preserving correspondence between visual evidence, tracking IDs and numerical results. A plausible overlay alone does not prove correct counting or runtime measurement.",
            ),
            (
                "p",
                "The implementation keeps processing logic independent of the interface. A deterministic generator supplies known object centers and classes. Both notebooks call the same modules as the app and camera entry point. The benchmark produces its own video, CSVs, stage figures and quality measurements. No detector was trained, no accuracy result was inferred from unlabeled footage, and no production-scale deployment is claimed.",
            ),
            ("image", ("docs/demo/architecture.png", 490, 164)),
            (
                "caption",
                "Figure 1. Two complementary vision paths share annotation, telemetry and artifact export. The classical path is the mandatory internship pipeline.",
            ),
            ("h", "Architecture and responsibilities"),
            (
                "p",
                "Small modules separate preprocessing, segmentation, tracking, optional pretrained inference, video orchestration and metrics. The application performs batch processing on demand; the optional camera script processes consecutive frames as they arrive. Frame selection reads saved images without repeating inference. This separation keeps the official pipeline easy to inspect and explain in an interview.",
            ),
        ],
    ),
    (
        "The required classical pipeline",
        [
            ("h", "From pixels to candidate shapes"),
            (
                "p",
                "Each decoded BGR frame is converted to grayscale. A 5 x 5 Gaussian filter with OpenCV-selected sigma suppresses high-frequency noise before thresholding. The app permits 3, 5 or 7 pixel kernels: increasing blur can remove noise but also smooth small corners and merge nearby boundaries. Gaussian filtering is performed on every processed frame, not only in a demonstration cell.",
            ),
            (
                "p",
                "Adaptive Gaussian thresholding uses a 31 x 31 local neighborhood and C=7. OpenCV subtracts C from the weighted neighborhood mean and applies inverse binary thresholding, selecting sufficiently dark pixels. Adaptive Mean is also exposed. Block size must be an odd integer at least 3. Local thresholds accommodate mild illumination variation, but texture can generate unwanted contours. Dark objects on a light background are an explicit default assumption.",
            ),
            (
                "p",
                "A 3 x 3 closing operation bridges small gaps, followed by opening to suppress isolated noise. Cleanup is configurable and can be disabled; it does not replace thresholding. The saved mask can contain boundary bands with hollow interiors. External contours recover enclosed boundaries; area filtering retains contours between 250 and 20,000 square pixels. This is contour segmentation, not a dense semantic mask.",
            ),
            ("image", ("outputs/charts/pipeline_stages.png", 490, 245)),
            (
                "caption",
                "Figure 2. Actual frame 120: original, grayscale, Gaussian blur, adaptive threshold, cleaned mask, contour overlay and tracked output.",
            ),
            ("h", "Geometry and pixel bounds"),
            (
                "p",
                "For each contour, the code computes area, closed perimeter, bounding rectangle and image moments. The centroid is (m10/m00, m01/m00); zero-area contours are rejected. Polygon approximation uses 2.5% of perimeter: three vertices indicate Triangle, four indicate Rectangle, and at least six plus circularity 4*pi*A/P^2 >= 0.76 indicate Circle. Remaining contours are Other. These are geometric labels, not general-purpose semantic classes.",
            ),
        ],
    ),
    (
        "Tracking, counting and frame analytics",
        [
            ("h", "Centroid association from scratch"),
            (
                "p",
                "The tracker enumerates Euclidean distances between retained track centroids and current detections, sorts candidate pairs, and accepts a pair only if neither member has been used. The default maximum distance is 55 pixels. Unmatched detections receive monotonically increasing integer IDs. Unmatched tracks accumulate missed frames and expire after more than 10 misses. A track can reconnect within that window near its last observed location.",
            ),
            (
                "p",
                "This is transparent greedy assignment, not a Kalman filter, global Hungarian assignment or appearance tracker. Fast motion, occlusion, intersecting trajectories and camera movement can cause identity swaps. Active tracks counts IDs observed in the current frame, excluding retained missing tracks. Unique tracks counts assigned IDs across the run and can exceed the number of physical objects.",
            ),
            ("h", "ROI, movement trails and line crossing"),
            (
                "p",
                "Each visible track keeps at most 24 recent coordinates. Analytics histories expire after 30 unseen frames. The ROI uses normalized numeric bounds and counts detections with centroids inside it. Full-frame processing remains active, so total and ROI counts are comparable. The benchmark ROI spans x=0.2-0.8 and y=0.1-0.9; it does not crop the input.",
            ),
            (
                "p",
                "A vertical line at x=0.5 counts left-to-right as Entered and right-to-left as Exited. A four-pixel deadband ignores uncertain positions near the line. Each ID is counted at most once per run, even if it moves back. These are image-plane directions, not a calibrated physical entrance. Fragmented IDs can still overcount a physical object; that limitation is disclosed.",
            ),
            ("image", ("outputs/frames/frame_0120_tracked.png", 420, 236.25)),
            (
                "caption",
                "Figure 3. Actual tracked frame with IDs, geometric labels, ROI, line, bounded trails and telemetry footer.",
            ),
            (
                "p",
                "The HUD reports mode, frame, objects, currently visible tracks, processing latency, processing FPS, ROI count, entered/exited totals and class counts. Counts are recalculated from detections every frame. No random IDs are substituted for actual tracker output.",
            ),
        ],
    ),
    (
        "Experimental setup and measured results",
        [
            ("h", "Reproducible evaluation protocol"),
            (
                "p",
                "NumPy seed 42 generates 240 frames at 30 source FPS (8 seconds), 960 x 540 resolution. Three shapes occupy separate lanes: a circle, rectangle and triangle. A smaller circle appears at frame 46 and disappears after frame 190. Mild Gaussian noise and a changing background gradient are included. Ground truth stores frame, object identity, shape and center. This deliberately easy, non-overlapping sequence does not represent natural-scene difficulty.",
            ),
            (
                "p",
                f"Execution used Python {run['python']}, OpenCV {run['opencv']}, CPU processing and {run['platform']}. Processor: {run['processor']}. Exact direct dependency pins and environment metadata are saved in the requirements files and outputs/run.json. Runtime varies with background load, codec behavior and OpenCV thread scheduling; the table represents one run.",
            ),
            ("table", summaryRows),
            ("h", "Timing definitions"),
            (
                "p",
                "A perf_counter interval starts after decode and covers preprocessing or inference, association, analytics and geometric overlays. It stops before the HUD that displays the result. Decode, video writing, FFmpeg conversion, disk exports and model load are excluded. Per-frame FPS equals 1000/processing_time_ms. Mean per-frame FPS averages these reciprocals; aggregate FPS is 1000 divided by mean latency. The latter represents throughput over the measured intervals. Model first-inference overhead remains included.",
            ),
            ("image", ("docs/demo/performance_summary.png", 490, 139)),
            (
                "caption",
                "Figure 4. Actual latency, processing-FPS and object-count series. Source playback stays at 30 FPS regardless of processing throughput.",
            ),
        ],
    ),
    (
        "Quality evaluation and deep vision",
        [
            ("h", "Known synthetic ground truth"),
            (
                "p",
                f"Each frame is matched by one-to-one nearest centroids within 20 pixels. Class labels are checked after spatial matching. Across {e['matched']} matched object-frame instances there were {e['false_positives']} unmatched predictions and {e['missed']} misses. Centroid precision and recall were both {e['centroid_precision']:.1%}; shape accuracy on matches was {e['shape_accuracy_on_matches']:.1%}. Mean center error was {e['mean_center_error_px']:.3f} pixels. Count MAE was {e['count_mae']:.3f}, with exact counts in {e['count_exact_frame_fraction']:.1%} of frames. There were {e['matched_track_id_changes']} matched ID changes and four assigned IDs.",
            ),
            (
                "p",
                "These values are valid for the separated lanes, where correspondence is unambiguous. They are not mAP, segmentation IoU, MOTA or evidence of generalization. The sequence does not test occlusion, clutter, polarity reversal or merged contours. Tests separately cover invalid settings, area bounds, matching uniqueness, stale tracks, bounded histories, line jitter, bad media and encoded-video readback.",
            ),
            ("h", "Pretrained boxes, instance masks and ByteTrack"),
            (
                "p",
                "The optional extension loads YOLOv8n for detection and YOLOv8n-seg for instance segmentation. Ultralytics runs on CPU with confidence 0.25 and inference size 640. Only the segmentation checkpoint supplies masks. Video calls use ByteTrack with persist=True; still-image inference does not invent IDs. A new model/tracker per run prevents identities from leaking between unrelated uploads.",
            ),
            (
                "p",
                f"Both models ran on the attributed Ultralytics bus photograph and a 20-frame translated-photo clip (480 x 640, 10 FPS). Detection returned {sum(yolo['detection']['image_class_counts'].values())} image boxes; segmentation returned {yolo['segmentation']['mask_count']} masks. Both videos exported and decoded correctly, with persistent IDs. This is an API integration smoke test, not natural-motion validation. The detection first frame took {yolo['detection']['summary']['maximum_processing_time_ms']:.1f} ms; excluding it would hide startup cost. No speed ranking against OpenCV is made because inputs and initialization conditions differ.",
            ),
            ("image", ("docs/demo/representative_segmentation.png", 165, 220)),
            (
                "caption",
                "Figure 5. Actual instance-mask output. Photograph: Ultralytics repository; see THIRD_PARTY_NOTICES.md.",
            ),
        ],
    ),
    (
        "Application, requirements and limitations",
        [
            ("h", "Inspection and export workflow"),
            (
                "p",
                "Streamlit offers Live Vision Pipeline, Frame Analysis, Performance Dashboard, Deep Vision, Export Center and About Project. Users process the demo or upload MP4, tune validated parameters and configure ROI and crossing analytics. The inspector reads three representative frame sets; selection changes do not rerun inference. The performance view loads real CSVs and run configuration.",
            ),
            (
                "p",
                "Uploads use generated local directories and fixed internal filenames. User runs are excluded from Git. Exports include video, frame and summary metrics, detections, metadata, images and charts. FFmpeg converts to H.264 when available; otherwise mp4v may not play in every browser. Output preserves dimensions and source FPS but omits audio. Capture and writer resources are released on errors.",
            ),
            ("table", mapping),
            ("h", "Additional engineering and remaining limits"),
            (
                "p",
                "Additions include bounded trails, ROI class counts, deadband line counting, ground truth, optional ByteTrack, instance masks, saved notebook outputs and browser-verified screenshots. Missing Ultralytics or weights leave the classical path available. The camera script handles an unavailable device; physical webcam operation still needs local hardware and permission verification.",
            ),
            (
                "p",
                "The classical path assumes dark foreground and needs tuning for scale or background changes. The tracker is unsuitable for dense crowds. No GPU acceleration or production deployment was evaluated. Runs persist locally until removed; large videos consume time and disk space. Cancellation and browser-camera streaming are not implemented. Controlled occlusion tests and labeled natural footage are needed before stronger quality claims.",
            ),
            ("h", "Conclusion and references"),
            (
                "p",
                "The official pipeline is visible in code, executable notebooks, stage figures and measured output. Synthetic evaluation establishes controlled geometric correctness; pretrained vision adds a separate capability without replacing required OpenCV stages. Every reported result is traceable to saved artifacts.",
            ),
            (
                "p",
                "References: docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html; docs.ultralytics.com/modes/track/; docs.ultralytics.com/models/yolov8/. Evidence: outputs/run.json, frame_metrics.csv, synthetic_evaluation.json and yolo_verification.json.",
            ),
        ],
    ),
]

styles = getSampleStyleSheet()
for name, size, leading, color in [
    ("BodyCustom", 9.3, 13, "#243247"),
    ("SectionCustom", 12, 15, "#126d68"),
    ("CaptionCustom", 8, 10, "#5b6b7c"),
    ("TitleCustom", 23, 27, "#101e30"),
    ("SubtitleCustom", 13, 17, "#126d68"),
    ("TableCustom", 8.2, 11, "#243247"),
]:
    styles.add(
        ParagraphStyle(
            name=name,
            fontName="Helvetica-Bold"
            if name in ("TitleCustom", "SectionCustom")
            else "Helvetica",
            fontSize=size,
            leading=leading,
            spaceAfter=8,
            textColor=colors.HexColor(color),
        )
    )


def paragraph(text, style):
    return Paragraph(html.escape(text), styles[style])


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor("#c5d3df"))
    canvas.line(42, 40, 553, 40)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5b6b7c"))
    canvas.drawString(42, 28, "VISION INTELLIGENCE | PROGREE TASK 4 | FAYZ LIAQAT")
    canvas.drawRightString(553, 28, str(doc.page))


def markdownTable(data):
    return (
        "| "
        + " | ".join(data[0])
        + " |\n|---|---|\n"
        + "\n".join("| " + " | ".join(row) + " |" for row in data[1:])
    )


story, markdown = [], []
for index, (title, blocks) in enumerate(pages):
    if index:
        story.append(PageBreak())
    story.append(paragraph(title, "TitleCustom"))
    markdown.append("# " + title + "\n")
    for kind, data in blocks:
        if kind == "table":
            table = Table(
                [[paragraph(str(v), "TableCustom") for v in row] for row in data],
                colWidths=[192, 298],
                hAlign="LEFT",
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcefea")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.HexColor("#f1f5f8"), colors.white],
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend([table, Spacer(1, 9)])
            markdown.append(markdownTable(data) + "\n")
        elif kind == "image":
            path, width, height = data
            story.append(
                Image(str(ROOT / path), width=width, height=height, hAlign="LEFT")
            )
            markdown.append(f"![{Path(path).stem}](../{path})\n")
        else:
            style = {
                "p": "BodyCustom",
                "h": "SectionCustom",
                "caption": "CaptionCustom",
                "subtitle": "SubtitleCustom",
            }[kind]
            story.append(paragraph(data, style))
            markdown.append(("## " if kind == "h" else "") + data + "\n")
    markdown.append("\n<!-- page break -->\n")
SimpleDocTemplate(
    str(ROOT / "report/Task_4_Whitepaper.pdf"),
    pagesize=A4,
    rightMargin=42,
    leftMargin=42,
    topMargin=38,
    bottomMargin=53,
    title="Vision Intelligence - Task 4 Whitepaper",
    author="Fayz Liaqat",
).build(story, onFirstPage=footer, onLaterPages=footer)
(ROOT / "report/Task_4_Whitepaper.md").write_text("\n".join(markdown), encoding="utf-8")
# Update the marked tables without changing the rest of the showcase README.
readmePath = ROOT / "README.md"
if readmePath.exists():
    readme = readmePath.read_text(encoding="utf-8")
    for marker, table in [("RESULTS", summaryRows), ("REQUIREMENTS", mapping)]:
        before, remainder = readme.split(f"<!-- {marker}_START -->")
        _, after = remainder.split(f"<!-- {marker}_END -->")
        readme = (
            before
            + f"<!-- {marker}_START -->\n"
            + markdownTable(table)
            + f"\n<!-- {marker}_END -->"
            + after
        )
    readmePath.write_text(readme, encoding="utf-8")
print("Built whitepaper and showcase figures from saved measurements")

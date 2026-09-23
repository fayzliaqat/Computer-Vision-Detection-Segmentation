# LinkedIn showcase edit

Decision: **C — short composite**, made locally from actual project outputs. Silent, 1920 × 1080, 30 FPS, H.264/yuv420p. The final media check records the exact duration and frame count in `media_validation.json`.

| Approximate timeline | Actual source | Text / purpose |
|---|---|---|
| 0–11 s | `outputs/traffic/videos/traffic_annotated.mp4`, from source time 3.93 s through the end | Turning traffic footage into vehicle-flow analytics. Boxes, classes, IDs, short trails, counts and per-frame telemetry are the recorded pipeline output. |
| 11–16 s | `pipeline_proof.png`, assembled from the four named `outputs/frames/frame_0120_*.png` stages | Original, Gaussian Blur, Adaptive Threshold, Contours + pixel bounds. This section is the controlled synthetic-shape example, not traffic accuracy evidence. |
| 16–20 s | `outputs/traffic_segmentation/videos/yolo_segmentation_output.mp4`, first four seconds | Instance segmentation / actual YOLOv8n-seg masks. Independently processed traffic footage; its counter and timings belong to that run. |
| 20–25 s | Actual `docs/demo/traffic_performance.png` Streamlit capture | Measured telemetry / actual Streamlit dashboard. Crop retains run metrics and latency/FPS charts. Peak vehicles is a visible count, not unique vehicles or congestion. |
| 25–28 s | Simple closing title | Vision Intelligence; OpenCV + YOLO + ByteTrack + Streamlit; Computer-Vision-Detection-Segmentation. |

Transitions are 0.35-second fades. Prepared segment durations are 11.35, 5.35, 4.35, 5.35 and 3 seconds; overlap and frame rounding determine the final duration. Source playback remains at its original pace. 25 FPS source frames are duplicated as needed for 30 FPS output, without generated motion. The segmentation ending holds briefly through its transition. No fabricated interface, detection footage, analytics or generated visuals are used.

## Rebuild locally

Install the project's core dependencies and FFmpeg with libx264/drawtext. Keep the committed source artifacts and dashboard capture at their original paths. From the repository root:

```powershell
python scripts/build_linkedin.py --prepare --compose
```

Preparation writes intermediate clips and title files into ignored `tmp/linkedin/`. The final output is `docs/linkedin/linkedin_demo.mp4`. The FFmpeg composition plugin's typed concat action was preflighted and executed, but readback found 4:4:4 pixels despite the requested 4:2:0 setting. The delivered video was rerendered using the portable native `--compose` path with explicit yuv420p, 0.35-second fades and no audio. Intermediate absolute-path requests remain local.

## Optional later editor polish

Use the five prepared `tmp/linkedin/part_*.mp4` clips in numeric order, with the durations and overlaps above. Keep all titles and factual source imagery. No external editor is required for the delivered video.

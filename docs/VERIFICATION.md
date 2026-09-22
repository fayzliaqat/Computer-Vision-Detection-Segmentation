# Verification record

Verified locally on Windows with Python 3.13.3. Canonical measurements are in `outputs/frame_metrics.csv`, `summary_metrics.csv`, `run.json` and `synthetic_evaluation.json`.

## Required checks

| Check | Evidence / result |
|---|---|
| Generate and decode synthetic video | 240 frames; 960 x 540; 30 FPS; actual OpenCV readback |
| Gaussian filtering | Runtime pipeline, stage PNGs and shape test |
| Adaptive thresholding | Runtime pipeline, binary-mask test and stage PNGs |
| Morphological cleanup | Closing/opening executed; saved cleaned mask |
| Contours and area bounds | Shape test verifies three classes and exclusion by area |
| Shape classification | Circle, Rectangle, Triangle and Other implemented; controlled matching evaluation |
| Centroid computation | Image moments; mean center error measured in synthetic_evaluation.json |
| ID assignment and persistence | One-to-one, persistence, expiry and bounded-trail tests pass |
| Dynamic counts | Actual class counts and HUD on every output frame |
| Annotated video | H.264 export, all 240 frames decoded after writing |
| Frame CSV | Exactly 240 rows; measured positive durations and reciprocal FPS checked |
| Pipeline images and charts | Actual early/middle/later frames; seven-stage figure; five charts |
| Notebook 1 | All six code cells executed; outputs retained; no error outputs |
| Notebook 2 | All four code cells executed; full video processed; outputs retained |
| Streamlit launch | Running on localhost:8504; actual browser screenshots saved |
| Built-in demo | Streamlit AppTest clicked Process video successfully |
| MP4 upload | Actual browser selected sample MP4 and completed input.mp4 run |
| Frame Analysis | AppTest and actual browser navigation; saved-stage inspection |
| Performance Dashboard | Actual run CSV; AppTest and browser screenshot |
| Export Center / About | AppTest navigated both without exceptions |
| YOLO detection | Real YOLOv8n image and 20-frame video inference; ByteTrack IDs |
| YOLO segmentation | Real YOLOv8n-seg masks and 20-frame video inference |
| Deep image UI | Actual browser detection and completed segmentation state verified |
| Webcam unavailable | Mocked test and actual camera index 999 both exit gracefully |
| Whitepaper | Exactly six PDF pages; all six rendered and visually inspected |
| Clean installation | New environment without system-site packages; requirements.txt installed; pip check passed |
| Full tests | All 10 unittest/AppTest tests passed in both working and clean core environments |
| Static checks | Ruff E9/F and Python compilation passed |
| Publication hygiene | Local links resolve; credential-pattern scan clear; all individual files under 10 MB |
| Isolation | Task 3 git status remains clean; no Task 3 files modified |

## Resolved development issues

- The initial AppTest addressed the main-area radio instead of sidebar navigation. The test selector was corrected; all pages now pass.
- Browser automation initially raced page navigation before uploading a deep image. Explicit page/result waits verified the completed detection and segmentation states.
- Initial Ultralytics settings fallback was fixed by creating its project-local ignored config directory before import.
- Screenshot review caught heading spacing, metric truncation and a text-encoding issue. These were corrected and actual screenshots recaptured.
- Windows notebook execution emitted a non-fatal ZMQ selector-thread warning. Both notebooks completed without error outputs. Streamlit AppTest emits its expected bare-mode context warning.

## Scope and manual checks

The physical webcam has not been tested: verify camera permissions and device selection locally. YOLO's translated-photo clip is an integration test, not a natural-motion benchmark. Synthetic correctness applies to separated geometric shapes. Test representative real footage before making broader accuracy or tracking claims. Videos omit audio. Local interactive runs persist until removed.

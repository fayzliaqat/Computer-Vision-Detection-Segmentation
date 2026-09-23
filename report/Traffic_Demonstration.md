# Real-world traffic demonstration

This addendum complements the unchanged six-page [synthetic whitepaper](Task_4_Whitepaper.pdf). It is a separate unlabeled real-world demonstration.

## Input and scene

User-provided `cars.mp4`: 1280 x 720, 25 source FPS, 382 frames, 15.28 seconds. A fixed elevated camera views both carriageways in daylight. Nearer vehicles are larger; distant cars are small, and poles intermittently occlude targets. No second traffic clip was added. Permission and provenance are recorded in [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) and [provenance.json](../outputs/traffic/provenance.json).

The originally supplied test file already contained model overlays. The clean source was recovered and verified against the user's newly supplied original before inference, avoiding detection over burned-in boxes or inherited metrics.

## Method

YOLOv8n on CPU, confidence 0.30, inference size 640; existing ByteTrack with persistent IDs. The preset restricts inference and analytics to car, truck, bus, motorcycle and bicycle. It does not reset IDs or change ByteTrack association. Trails retain 12 points. Labels omit confidence values and are suppressed for very small distant boxes; those detections still contribute to counts.

A dashed horizontal line is placed at 55% of source height (y=396 pixels), across the observed direction of road travel. Direction A is top-to-bottom; Direction B is bottom-to-top. A 4-pixel deadband suppresses line jitter. Each track ID can contribute once per run. Events log timestamp, frame, ID, class at crossing and centroid for inspection. ROI behavior and legacy vertical counting remain compatible with the classical pipeline.

## Measured results

| Metric | Observed value |
|---|---|
| Frames processed | 382 |
| Mean processing latency | 45.931 ms |
| Mean inference call time | 41.799 ms |
| Aggregate processing throughput | 21.772 FPS |
| Mean per-frame processing FPS | 26.256 |
| P50 / P95 latency | 37.644 / 41.732 ms |
| First-frame processing | 3020.296 ms |
| Mean / peak active vehicles | 20.589 / 26 |
| Total crossings | 10 |
| Direction A / B | 2 / 8 |

Source: [frame metrics](../outputs/traffic/frame_metrics.csv), [run metadata](../outputs/traffic/run.json), [crossing events](../outputs/traffic/crossing_events.csv). Processing includes inference, analytics and overlays; excludes decoding, the telemetry HUD, encoding, model loading and disk writes. First-inference initialization remains included. Video playback retains 25 FPS, irrespective of the processing throughput. This is batch CPU analysis, not a measured live deployment.

| Vehicle class | Mean visible per frame | Peak visible |
|---|---:|---:|
| car | 17.079 | 21 |
| truck | 3.081 | 7 |
| bus | 0.429 | 3 |
| motorcycle | 0.000 | 0 |
| bicycle | 0.000 | 0 |

No motorcycles or bicycles were detected in this run. These per-frame counts must not be summed and called unique physical vehicles. Labels can fluctuate between visually similar large vehicles; a bus/truck class is not verified ground truth.

## Inspection and limits

Beginning, middle and final annotated frames and the ten logged crossing frames were inspected. The previous vertical divider line was unsuitable for interpreting travel direction. Horizontal counting and neutral labels correct that presentation issue. The new HUD is appended below the source image (annotated output 1280 x 802), preserving the entire scene. Labels are shorter and collision-aware; trails and line thickness are restrained.

No association rewrite or ID renumbering was made. Distant detections can fragment, and poles can interrupt tracks; stable IDs around an observed crossing do not establish general tracking accuracy. There are no manual ground-truth labels, so no precision, recall, mAP or crossing accuracy is claimed. Class confusion, missed detections and fragmentation can affect results on other clips. Visible vehicle count is not a defined congestion or density metric.

YOLOv8n-seg was separately executed on the first 100 frames (4 seconds) of this same clean source and exported actual instance masks. It is visual integration evidence, not a segmentation-accuracy benchmark or a like-for-like speed ranking.

The original classical benchmark, Gaussian/adaptive stages, centroid tracker, notebooks and six-page whitepaper are preserved.

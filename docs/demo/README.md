# Showcase recording guide

Record the actual local app at a desktop resolution. Use the existing generated artifacts; no screen mocks are needed.

1. **0-8 seconds:** show the landing page and play the annotated shapes demo. Point out persistent IDs and dynamic class counts.
2. **8-20 seconds:** open Frame Analysis and compare grayscale, blur, adaptive threshold and contour output.
3. **20-30 seconds:** open Performance Dashboard; explain source FPS versus processing FPS and P95 latency.
4. **30-45 seconds:** open Deep Vision and upload data/bus.jpg for actual detection or instance masks. Mention pretrained YOLO explicitly.
5. **45-55 seconds:** show Export Center and the six-page whitepaper.

Do not describe the translated bus-photo smoke test as real traffic tracking. Do not quote synthetic correctness as general-world detection accuracy. Webcam operation depends on local hardware; the browser app processes uploaded video on demand.

## Included assets

- streamlit_dashboard.png, streamlit_inspector.png, streamlit_performance.png, streamlit_deep_vision.png: actual running app captures.
- architecture.png: implementation architecture.
- pipeline_stages.png: actual seven-stage figure.
- performance_summary.png: canonical measured telemetry.
- representative_classical.png: actual tracked synthetic frame.
- representative_detection.png and representative_segmentation.png: actual model outputs.

No content is automatically posted to any platform.

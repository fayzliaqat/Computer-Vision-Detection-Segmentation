Traffic cameras generate hours of footage, but raw video alone doesn't tell you how many vehicles crossed a monitoring line, what classes were visible, or how movement changed over time.

I built Vision Intelligence to turn video into tracked detections, vehicle counts, directional crossing events and exportable frame-level telemetry. Traffic-flow analytics is its main real-world showcase.

The foundation is my Progree AI internship's classical OpenCV pipeline: grayscale, Gaussian blur, adaptive thresholding, morphological cleanup, contours, pixel-area filtering, shape classification, tracking and dynamic counts. A Streamlit inspector makes each stage visible.

I extended it with YOLOv8 detection, actual instance segmentation masks and ByteTrack. The traffic preset focuses on cars, trucks, buses, motorcycles and bicycles, with short movement trails and a horizontal monitoring line suited to this scene.

On my 15.28-second traffic clip (382 frames at 25 FPS), the CPU run recorded 10 line crossings: 2 in Direction A and 8 in Direction B. Mean processing latency was 45.93 ms, equivalent to 21.77 processing FPS. That includes first-inference startup and excludes decoding, video encoding and disk I/O; it isn't end-to-end deployment speed.

This demonstrates automated traffic-flow analytics from video. The footage has no ground-truth labels, so I don't claim real-world accuracy or congestion measurement. Small, occluded vehicles and class changes remain limitations.

Code, measured results, annotated video, CSV exports and the six-page whitepaper:
https://github.com/fayzliaqat/Computer-Vision-Detection-Segmentation

#ComputerVision #OpenCV #MachineLearning

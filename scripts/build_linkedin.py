"""Prepare a factual 28-second showcase; optional portable native FFmpeg composition.

python scripts/build_linkedin.py --prepare
python scripts/build_linkedin.py --compose

The typed composition tool was preflighted and tested, but its output was 4:4:4.
The final --compose render explicitly selects 4:2:0 for browser compatibility.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / "tmp/linkedin"
OUT = ROOT / "docs/linkedin"
DURATIONS = [11.35, 5.35, 4.35, 5.35, 3.0]


def execute(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])


def findFont():
    candidates = [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise RuntimeError("Install Segoe UI or DejaVu Sans for title rendering")


def prepare():
    TEMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fontPath = findFont()
    shutil.copyfile(fontPath, TEMP / "font.ttf")
    titleFont = ImageFont.truetype(str(fontPath), 43)
    smallFont = ImageFont.truetype(str(fontPath), 28)
    background = "#0b1220"
    board = Image.new("RGB", (1920, 1080), background)
    draw = ImageDraw.Draw(board)
    draw.text(
        (95, 35),
        "Classical vision | the required processing stages",
        font=titleFont,
        fill="#e5edf7",
    )
    for name, label, x, y in [
        ("original", "Original", 130, 105),
        ("blurred", "Gaussian Blur", 1030, 105),
        ("threshold", "Adaptive Threshold", 130, 590),
        ("segmented", "Contours + pixel bounds", 1030, 590),
    ]:
        picture = Image.open(ROOT / f"outputs/frames/frame_0120_{name}.png").convert(
            "RGB"
        )
        picture = ImageOps.contain(picture, (760, 427))
        draw.text((x, y), label, font=smallFont, fill="#98c5c2")
        board.paste(picture, (x, y + 42))
    board.save(OUT / "pipeline_proof.png")
    end = Image.new("RGB", (1920, 1080), background)
    draw = ImageDraw.Draw(end)
    draw.line((150, 360, 1770, 360), fill="#689d9d", width=3)
    for text, y, font in [
        ("Vision Intelligence", 415, ImageFont.truetype(str(fontPath), 70)),
        ("OpenCV + YOLO + ByteTrack + Streamlit", 535, titleFont),
        ("Computer-Vision-Detection-Segmentation", 635, smallFont),
        ("github.com/fayzliaqat", 697, smallFont),
    ]:
        width = draw.textbbox((0, 0), text, font=font)[2]
        draw.text(((1920 - width) // 2, y), text, font=font, fill="#e5edf7")
    end.save(TEMP / "closing.png")

    assets = [
        ROOT / "outputs/traffic/videos/traffic_annotated.mp4",
        OUT / "pipeline_proof.png",
        ROOT / "outputs/traffic_segmentation/videos/yolo_segmentation_output.mp4",
        ROOT / "docs/demo/traffic_performance.png",
        TEMP / "closing.png",
    ]
    titles = [
        "Turning traffic footage into vehicle-flow analytics",
        "",
        "Instance segmentation | actual YOLOv8n-seg masks",
        "Measured telemetry | actual Streamlit dashboard",
        "",
    ]
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required")
    for index, (source, duration, title) in enumerate(zip(assets, DURATIONS, titles)):
        if not source.exists():
            raise FileNotFoundError(source)
        args = [ffmpeg, "-y", "-loglevel", "error"]
        if index == 0:
            args += ["-ss", "3.93"]
        if source.suffix == ".png":
            args += ["-loop", "1", "-framerate", "30"]
        args += ["-i", str(source)]
        if title:
            textPath = TEMP / f"title_{index}.txt"
            textPath.write_text(title, encoding="utf-8")
            # Crop the actual dashboard capture to its metrics and first two charts.
            filters = "crop=1320:630:455:250," if index == 3 else ""
            filters += "scale=1880:930:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=1920:1080:(ow-iw)/2:125:color=0x0b1220"
            filters += f",drawtext=fontfile=tmp/linkedin/font.ttf:textfile=tmp/linkedin/title_{index}.txt:fontsize=42:fontcolor=0xe5edf7:x=(w-text_w)/2:y=38"
        else:
            filters = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x0b1220"
        filters += (
            ",fps=30,format=yuv420p,setsar=1,tpad=stop_mode=clone:stop_duration=1"
        )
        args += [
            "-vf",
            filters,
            "-t",
            str(duration),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "19",
            "-movflags",
            "+faststart",
            str(TEMP / f"part_{index}.mp4"),
        ]
        execute(args)
        print(f"Prepared part {index + 1}/5", flush=True)
    request = {
        "context": "codex",
        "cwd": str(ROOT),
        "input": {
            "action": "concat",
            "inputs": [str(TEMP / f"part_{i}.mp4") for i in range(5)],
            "transition": "fade",
            "transitionDuration": 0.35,
            "audio": "drop",
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "pixelFormat": "yuv420p",
            "hardware": "software",
            "output": str(OUT / "linkedin_demo.mp4"),
            "overwrite": True,
        },
    }
    (TEMP / "composition.json").write_text(
        json.dumps(request, indent=2), encoding="utf-8"
    )


def compose():
    ffmpeg = shutil.which("ffmpeg")
    args = [ffmpeg, "-y", "-loglevel", "error", "-filter_complex_threads", "1"]
    for i in range(5):
        args += ["-i", str(TEMP / f"part_{i}.mp4")]
    filters = [
        f"[{i}:v]setpts=PTS-STARTPTS,fps=30,format=yuv420p,settb=AVTB[v{i}]"
        for i in range(5)
    ]
    previous = "v0"
    duration = DURATIONS[0]
    for i in range(1, 5):
        filters.append(
            f"[{previous}][v{i}]xfade=transition=fade:duration=0.35:offset={duration - 0.35:.3f}[mix{i}]"
        )
        duration += DURATIONS[i] - 0.35
        previous = f"mix{i}"
    args += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        f"[{previous}]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(OUT / "linkedin_demo.mp4"),
    ]
    execute(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--compose", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.compose:
        compose()
    if not args.prepare and not args.compose:
        parser.error("Choose --prepare and/or --compose")

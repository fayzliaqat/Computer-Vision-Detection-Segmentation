"""Generate, process and evaluate the controlled benchmark. No real-world accuracy claims."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from generate_demo import generateDemo
from src.video_processor import processVideo

ROOT = Path(__file__).resolve().parent


def evaluateSynthetic(outputDir=ROOT / "outputs"):
    outputDir = Path(outputDir)
    truth = pd.read_csv(ROOT / "data" / "ground_truth.csv")
    predicted = pd.read_csv(outputDir / "detections.csv")
    metrics = pd.read_csv(outputDir / "frame_metrics.csv")
    matched = correct = falsePositive = missed = switches = 0
    distances, previousIds = [], {}
    for number in metrics.frame_number:
        gt = truth[truth.frame_number == number].to_dict("records")
        det = predicted[predicted.frame_number == number].to_dict("records")
        candidates = sorted(
            (
                float(
                    np.hypot(
                        g["center_x"] - d["center_x"], g["center_y"] - d["center_y"]
                    )
                ),
                i,
                j,
            )
            for i, g in enumerate(gt)
            for j, d in enumerate(det)
        )
        usedG, usedD = set(), set()
        for distance, i, j in candidates:
            if distance > 20:
                break
            if i in usedG or j in usedD:
                continue
            usedG.add(i)
            usedD.add(j)
            matched += 1
            correct += gt[i]["shape"] == det[j]["label"]
            distances.append(distance)
            identity = gt[i]["object_id"]
            if identity in previousIds and previousIds[identity] != det[j]["track_id"]:
                switches += 1
            previousIds[identity] = det[j]["track_id"]
        falsePositive += len(det) - len(usedD)
        missed += len(gt) - len(usedG)
    gtCounts = (
        truth.groupby("frame_number")
        .size()
        .reindex(metrics.frame_number, fill_value=0)
        .to_numpy()
    )
    errors = np.abs(metrics.objects_detected.to_numpy() - gtCounts)
    result = {
        "matching_rule": "Greedy one-to-one nearest centroid <= 20 pixels; controlled non-overlapping lanes",
        "matched": matched,
        "false_positives": falsePositive,
        "missed": missed,
        "centroid_precision": matched / (matched + falsePositive)
        if matched + falsePositive
        else 0,
        "centroid_recall": matched / (matched + missed) if matched + missed else 0,
        "shape_accuracy_on_matches": correct / matched if matched else 0,
        "mean_center_error_px": float(np.mean(distances)) if distances else None,
        "count_mae": float(errors.mean()),
        "count_exact_frame_fraction": float((errors == 0).mean()),
        "matched_track_id_changes": switches,
        "ground_truth_objects": int(truth.object_id.nunique()),
        "scope": "Synthetic benchmark only; not mAP, MOT accuracy or general-purpose CV accuracy.",
    }
    (outputDir / "synthetic_evaluation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    generateDemo()
    run = processVideo(
        ROOT / "data" / "sample_shapes.mp4",
        ROOT / "outputs",
        roi=(0.2, 0.1, 0.8, 0.9),
        line=0.5,
    )
    print(json.dumps(run["summary"], indent=2))
    print(json.dumps(evaluateSynthetic(), indent=2))

#!/usr/bin/env python3
"""
Analyze temporal track consistency using saved per-frame detection embeddings.

Example:
python analyze_track_embedding_consistency.py \
    --detections_dir outputs/video_embeddings \
    --tracks_txt path/to/tracks.txt \
    --output_dir outputs/track_embedding_analysis \
    --iou_thr 0.30 \
    --min_track_len 3

Primary strategy:
- Match each GT track box to the single best-IoU detection in the same frame.
- Reject matches with IoU < threshold.
- For a valid track at frame t, use the same track's matched embedding at frame t-1
  as the reference embedding.
- Compare that reference embedding against every detection embedding at frame t.
- Report whether the GT-matched detection at frame t is also the strongest cosine
  match for the previous-frame reference.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_int(value: str) -> Optional[int]:
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def compute_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a.astype(float)
    bx1, by1, bx2, by2 = box_b.astype(float)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_similarities_to_candidates(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if candidates.size == 0:
        return np.zeros((0,), dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    ref_norm = np.linalg.norm(ref)
    if ref_norm <= 1e-12:
        return np.zeros((candidates.shape[0],), dtype=np.float32)
    ref = ref / ref_norm
    cand = np.asarray(candidates, dtype=np.float32)
    cand_norms = np.linalg.norm(cand, axis=1, keepdims=True)
    cand = cand / np.clip(cand_norms, 1e-12, None)
    return (cand @ ref).astype(np.float32)


def load_tracks_txt(txt_path: Path):
    tracks = defaultdict(list)
    with txt_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            fr = safe_int(parts[0])
            tid = safe_int(parts[1])
            x = safe_int(parts[2])
            y = safe_int(parts[3])
            w = safe_int(parts[4])
            h = safe_int(parts[5])
            if None in (fr, tid, x, y, w, h):
                continue
            tracks[int(fr)].append((int(tid), int(x), int(y), int(w), int(h)))
    return tracks


def load_frame_detection_file(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {
        "frame_idx": np.array(data["frame_idx"]).astype(np.int32),
        "boxes_xyxy_abs": np.array(data["boxes_xyxy_abs"]).astype(np.float32),
        "scores": np.array(data["scores"]).astype(np.float32),
        "labels": np.array(data["labels"]).astype(np.int32),
        "label_names": np.array(data["label_names"]),
        "embeddings": np.array(data["embeddings"]).astype(np.float32),
        "detection_ids": np.array(data["detection_ids"]).astype(np.int32),
    }


def match_tracks_to_detections(
    detections_by_frame: Dict[int, Dict[str, np.ndarray]],
    tracks_by_frame: Dict[int, List[Tuple[int, int, int, int, int]]],
    iou_thr: float,
) -> List[Dict[str, object]]:
    """Best-IoU per track occurrence in same frame (not global 1:1 assignment)."""
    rows: List[Dict[str, object]] = []

    for frame_idx, track_rows in tracks_by_frame.items():
        frame_det = detections_by_frame.get(frame_idx)
        if frame_det is None:
            continue

        det_boxes = frame_det["boxes_xyxy_abs"]
        det_scores = frame_det["scores"]
        det_labels = frame_det["labels"]
        det_label_names = frame_det["label_names"]
        det_embeddings = frame_det["embeddings"]
        det_ids = frame_det["detection_ids"]

        if det_boxes.size == 0:
            continue

        for tid, x, y, w, h in track_rows:
            track_box = np.array([x, y, x + w, y + h], dtype=np.float32)
            ious = np.array([compute_iou_xyxy(track_box, db) for db in det_boxes], dtype=np.float32)
            best_idx = int(np.argmax(ious)) if ious.size > 0 else -1
            best_iou = float(ious[best_idx]) if best_idx >= 0 else 0.0
            if best_idx < 0 or best_iou < iou_thr:
                continue

            rows.append({
                "frame_idx": int(frame_idx),
                "track_id": int(tid),
                "track_x": float(x),
                "track_y": float(y),
                "track_w": float(w),
                "track_h": float(h),
                "track_x1": float(track_box[0]),
                "track_y1": float(track_box[1]),
                "track_x2": float(track_box[2]),
                "track_y2": float(track_box[3]),
                "det_idx_in_frame": int(best_idx),
                "detection_id": int(det_ids[best_idx]),
                "det_x1": float(det_boxes[best_idx][0]),
                "det_y1": float(det_boxes[best_idx][1]),
                "det_x2": float(det_boxes[best_idx][2]),
                "det_y2": float(det_boxes[best_idx][3]),
                "iou": best_iou,
                "score": float(det_scores[best_idx]),
                "label": int(det_labels[best_idx]),
                "label_name": str(det_label_names[best_idx]),
                "embedding": det_embeddings[best_idx].copy(),
            })
    return rows


def build_temporal_comparison_rows(
    matches_df: pd.DataFrame,
    detections_by_frame: Dict[int, Dict[str, np.ndarray]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for track_id, group in matches_df.groupby("track_id", sort=True):
        group = group.sort_values("frame_idx").reset_index(drop=True)
        for i in range(1, len(group)):
            prev_row = group.iloc[i - 1]
            curr_row = group.iloc[i]

            frame_t_minus_1 = int(prev_row["frame_idx"])
            frame_t = int(curr_row["frame_idx"])
            if frame_t != frame_t_minus_1 + 1:
                continue

            frame_det = detections_by_frame.get(frame_t)
            if frame_det is None:
                continue

            candidate_embeddings = frame_det["embeddings"]
            if candidate_embeddings.size == 0:
                continue

            matched_idx = int(curr_row["det_idx_in_frame"])
            if matched_idx < 0 or matched_idx >= candidate_embeddings.shape[0]:
                continue

            similarities = cosine_similarities_to_candidates(
                np.asarray(prev_row["embedding"], dtype=np.float32),
                candidate_embeddings,
            )
            if similarities.size == 0:
                continue

            matched_cosine = float(similarities[matched_idx])
            best_idx = int(np.argmax(similarities))
            best_cosine = float(similarities[best_idx])
            matched_rank = int(1 + np.count_nonzero(similarities > matched_cosine))

            rows.append({
                "track_id": int(track_id),
                "frame_t_minus_1": frame_t_minus_1,
                "frame_t": frame_t,
                "matched_cosine": matched_cosine,
                "matched_rank": matched_rank,
                "best_cosine": best_cosine,
                "is_top1_match": bool(matched_rank == 1),
                "num_candidates_t": int(similarities.shape[0]),
                "matched_det_idx_t": matched_idx,
                "best_det_idx_t": best_idx,
                "prev_detection_id": int(prev_row["detection_id"]),
                "matched_detection_id_t": int(curr_row["detection_id"]),
                "prev_iou": float(prev_row["iou"]),
                "matched_iou_t": float(curr_row["iou"]),
                "prev_score": float(prev_row["score"]),
                "matched_score_t": float(curr_row["score"]),
                "matched_label_t": int(curr_row["label"]),
                "matched_label_name_t": str(curr_row["label_name"]),
            })

    return rows


def summarize_temporal_track(
    temporal_df: pd.DataFrame,
    matched_track_length: int,
) -> Dict[str, object]:
    matched_cosines = temporal_df["matched_cosine"].to_numpy(dtype=np.float32)
    matched_ranks = temporal_df["matched_rank"].to_numpy(dtype=np.float32)
    best_cosines = temporal_df["best_cosine"].to_numpy(dtype=np.float32)
    is_top1 = temporal_df["is_top1_match"].to_numpy(dtype=np.float32)

    return {
        "track_id": int(temporal_df["track_id"].iloc[0]),
        "matched_track_length": int(matched_track_length),
        "num_temporal_pairs": int(len(temporal_df)),
        "matched_cosine_mean": float(np.mean(matched_cosines)) if matched_cosines.size else np.nan,
        "matched_cosine_std": float(np.std(matched_cosines)) if matched_cosines.size else np.nan,
        "matched_cosine_min": float(np.min(matched_cosines)) if matched_cosines.size else np.nan,
        "matched_cosine_max": float(np.max(matched_cosines)) if matched_cosines.size else np.nan,
        "matched_cosine_median": float(np.median(matched_cosines)) if matched_cosines.size else np.nan,
        "best_cosine_mean": float(np.mean(best_cosines)) if best_cosines.size else np.nan,
        "best_cosine_std": float(np.std(best_cosines)) if best_cosines.size else np.nan,
        "matched_rank_mean": float(np.mean(matched_ranks)) if matched_ranks.size else np.nan,
        "matched_rank_median": float(np.median(matched_ranks)) if matched_ranks.size else np.nan,
        "top1_rate": float(np.mean(is_top1)) if is_top1.size else np.nan,
        "top1_count": int(np.sum(is_top1)) if is_top1.size else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze track embedding consistency")
    parser.add_argument("--detections_dir", type=Path, required=True)
    parser.add_argument("--tracks_txt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--iou_thr", type=float, default=0.30)
    parser.add_argument("--min_track_len", type=int, default=3)
    parser.add_argument("--top_n_plots", type=int, default=20, help="Max track count for per-track plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_track").mkdir(exist_ok=True)
    (args.output_dir / "plots").mkdir(exist_ok=True)

    det_files = sorted(args.detections_dir.glob("frame_*.npz"))
    if not det_files:
        raise FileNotFoundError(f"No frame_*.npz found in {args.detections_dir}")

    detections_by_frame: Dict[int, Dict[str, np.ndarray]] = {}
    for p in det_files:
        d = load_frame_detection_file(p)
        frame_idx = int(np.asarray(d["frame_idx"]).reshape(-1)[0])
        detections_by_frame[frame_idx] = d

    tracks_by_frame = load_tracks_txt(args.tracks_txt)

    match_rows = match_tracks_to_detections(detections_by_frame, tracks_by_frame, args.iou_thr)
    if not match_rows:
        print("No matches found above IoU threshold.")
        empty_summary = args.output_dir / "track_summary.csv"
        empty_matches = args.output_dir / "track_matches.csv"
        empty_temporal = args.output_dir / "temporal_comparisons.csv"
        pd.DataFrame().to_csv(empty_summary, index=False)
        pd.DataFrame().to_csv(empty_matches, index=False)
        pd.DataFrame().to_csv(empty_temporal, index=False)
        return

    matches_df = pd.DataFrame(match_rows)
    matches_df = matches_df.sort_values(["track_id", "frame_idx"]).reset_index(drop=True)

    # Save GT-to-detection matches without raw embedding vectors.
    matches_csv = matches_df.drop(columns=["embedding"]).copy()
    matches_csv.to_csv(args.output_dir / "track_matches.csv", index=False)

    temporal_rows = build_temporal_comparison_rows(matches_df, detections_by_frame)
    temporal_csv_path = args.output_dir / "temporal_comparisons.csv"
    if not temporal_rows:
        pd.DataFrame().to_csv(temporal_csv_path, index=False)
        pd.DataFrame().to_csv(args.output_dir / "track_summary.csv", index=False)
        report = {
            "detections_dir": str(args.detections_dir),
            "tracks_txt": str(args.tracks_txt),
            "iou_threshold": float(args.iou_thr),
            "min_track_len": int(args.min_track_len),
            "num_detection_files": len(det_files),
            "num_frames_with_tracks": len(tracks_by_frame),
            "num_matches": int(len(matches_df)),
            "num_tracks_matched": int(matches_df["track_id"].nunique()),
            "num_temporal_comparisons": 0,
            "num_tracks_with_temporal_comparisons": 0,
            "global_top1_rate": np.nan,
            "global_matched_cosine_mean": np.nan,
            "global_rank_mean": np.nan,
        }
        with (args.output_dir / "analysis_report.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(report.keys()))
            writer.writeheader()
            writer.writerow(report)
        print("No valid consecutive t-1 -> t comparisons found after GT matching.")
        return

    temporal_df = pd.DataFrame(temporal_rows)
    temporal_df = temporal_df.sort_values(["track_id", "frame_t"]).reset_index(drop=True)
    temporal_df.to_csv(temporal_csv_path, index=False)

    matched_track_lengths = matches_df.groupby("track_id").size().to_dict()
    summary_rows = []
    all_matched_cosines: List[float] = []

    for track_id, group in temporal_df.groupby("track_id", sort=True):
        matched_track_length = int(matched_track_lengths.get(int(track_id), 0))
        if matched_track_length < args.min_track_len:
            continue
        summary = summarize_temporal_track(group, matched_track_length)
        summary_rows.append(summary)
        track_dir = args.output_dir / "per_track"
        group.to_csv(track_dir / f"track_{int(track_id):06d}_temporal.csv", index=False)
        all_matched_cosines.extend(group["matched_cosine"].tolist())

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_dir / "track_summary.csv", index=False)

    # Global histogram
    if all_matched_cosines:
        plt.figure(figsize=(8, 5))
        plt.hist(all_matched_cosines, bins=30)
        plt.title("Global matched cosine similarity for t-1 -> t")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(args.output_dir / "plots" / "global_hist_matched_cosine_t_minus_1_to_t.png", dpi=160)
        plt.close()

    # Boxplot by track id
    if summary_rows:
        top_track_ids = summary_df.sort_values("num_temporal_pairs", ascending=False)["track_id"].head(args.top_n_plots).astype(int).tolist()
        data = []
        labels = []
        for tid in top_track_ids:
            g = temporal_df[temporal_df["track_id"] == tid].sort_values("frame_t")
            data.append(g["matched_cosine"].tolist())
            labels.append(str(tid))

        if data:
            plt.figure(figsize=(max(8, len(data) * 0.45), 5))
            plt.boxplot(data, labels=labels, showfliers=False)
            plt.title("Matched cosine similarity by track for t-1 -> t")
            plt.xlabel("Track ID")
            plt.ylabel("Cosine similarity")
            plt.tight_layout()
            plt.savefig(args.output_dir / "plots" / "boxplot_matched_cosine_by_track.png", dpi=160)
            plt.close()

    # Per-track plots for top-N tracks with the most consecutive comparisons.
    for tid in summary_df.sort_values("num_temporal_pairs", ascending=False)["track_id"].head(args.top_n_plots).astype(int).tolist() if not summary_df.empty else []:
        g = temporal_df[temporal_df["track_id"] == tid].sort_values("frame_t")
        matched_cosine = g["matched_cosine"].to_numpy(dtype=np.float32)
        frames_t = g["frame_t"].to_numpy(dtype=np.int32)

        # Histogram per track
        plt.figure(figsize=(7, 4))
        plt.hist(matched_cosine, bins=20)
        plt.title(f"Track {tid} - matched cosine histogram for t-1 -> t")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(args.output_dir / "plots" / f"track_{tid:06d}_matched_cosine_hist.png", dpi=160)
        plt.close()

        # Timeline plot
        plt.figure(figsize=(8, 4))
        plt.plot(frames_t, matched_cosine, marker="o", linewidth=1)
        plt.title(f"Track {tid} - matched cosine over time")
        plt.xlabel("Frame t")
        plt.ylabel("Cosine similarity")
        plt.tight_layout()
        plt.savefig(args.output_dir / "plots" / f"track_{tid:06d}_matched_cosine_timeline.png", dpi=160)
        plt.close()

    # Save a brief run report
    report = {
        "detections_dir": str(args.detections_dir),
        "tracks_txt": str(args.tracks_txt),
        "iou_threshold": float(args.iou_thr),
        "min_track_len": int(args.min_track_len),
        "num_detection_files": len(det_files),
        "num_frames_with_tracks": len(tracks_by_frame),
        "num_matches": int(len(matches_df)),
        "num_tracks_matched": int(matches_df["track_id"].nunique()) if not matches_df.empty else 0,
        "num_temporal_comparisons": int(len(temporal_df)),
        "num_tracks_with_temporal_comparisons": int(temporal_df["track_id"].nunique()) if not temporal_df.empty else 0,
        "num_tracks_in_summary": int(len(summary_df)),
        "global_top1_rate": float(temporal_df["is_top1_match"].mean()) if not temporal_df.empty else np.nan,
        "global_matched_cosine_mean": float(temporal_df["matched_cosine"].mean()) if not temporal_df.empty else np.nan,
        "global_rank_mean": float(temporal_df["matched_rank"].mean()) if not temporal_df.empty else np.nan,
    }
    with (args.output_dir / "analysis_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report.keys()))
        writer.writeheader()
        writer.writerow(report)

    print(f"Analysis done. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

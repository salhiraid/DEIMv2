#!/usr/bin/env python3
"""
Analyze tracking consistency using saved per-frame detection embeddings.

Example:
python analyze_track_embedding_consistency.py \
    --detections_dir outputs/video_embeddings \
    --tracks_txt path/to/tracks.txt \
    --output_dir outputs/track_embedding_analysis \
    --iou_thr 0.30 \
    --min_track_len 3

Strategy:
- Match each track box to the single best-IoU detection in the same frame.
- Reject matches with IoU < threshold.
- Compute cosine similarity stability metrics per track.
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


def pairwise_cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb = embeddings / np.clip(norms, 1e-12, None)
    return (emb @ emb.T).astype(np.float32)


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


def summarize_track(track_df: pd.DataFrame) -> Dict[str, object]:
    track_df = track_df.sort_values("frame_idx").reset_index(drop=True)
    embeddings = np.stack(track_df["embedding"].to_numpy(), axis=0).astype(np.float32)

    ref = embeddings[0]
    sim_to_first = np.array([cosine_similarity(e, ref) for e in embeddings], dtype=np.float32)

    if len(embeddings) >= 2:
        cons_sim = np.array([
            cosine_similarity(embeddings[i], embeddings[i - 1])
            for i in range(1, len(embeddings))
        ], dtype=np.float32)
    else:
        cons_sim = np.zeros((0,), dtype=np.float32)

    pairwise = pairwise_cosine_matrix(embeddings)
    upper_vals = pairwise[np.triu_indices(pairwise.shape[0], k=1)] if pairwise.shape[0] > 1 else np.zeros((0,), dtype=np.float32)

    stats_source = sim_to_first
    return {
        "track_id": int(track_df["track_id"].iloc[0]),
        "track_length": int(len(track_df)),
        "num_matched": int(len(track_df)),
        "sim_to_first_mean": float(np.mean(stats_source)) if stats_source.size else np.nan,
        "sim_to_first_std": float(np.std(stats_source)) if stats_source.size else np.nan,
        "sim_to_first_var": float(np.var(stats_source)) if stats_source.size else np.nan,
        "sim_to_first_min": float(np.min(stats_source)) if stats_source.size else np.nan,
        "sim_to_first_max": float(np.max(stats_source)) if stats_source.size else np.nan,
        "sim_to_first_median": float(np.median(stats_source)) if stats_source.size else np.nan,
        "consecutive_mean": float(np.mean(cons_sim)) if cons_sim.size else np.nan,
        "consecutive_std": float(np.std(cons_sim)) if cons_sim.size else np.nan,
        "pairwise_mean": float(np.mean(upper_vals)) if upper_vals.size else np.nan,
        "pairwise_std": float(np.std(upper_vals)) if upper_vals.size else np.nan,
        "embeddings": embeddings,
        "sim_to_first": sim_to_first,
        "consecutive_sim": cons_sim,
        "pairwise_matrix": pairwise,
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
        pd.DataFrame().to_csv(empty_summary, index=False)
        pd.DataFrame().to_csv(empty_matches, index=False)
        return

    matches_df = pd.DataFrame(match_rows)
    matches_df = matches_df.sort_values(["track_id", "frame_idx"]).reset_index(drop=True)

    # Add cosine against first and previous within each track to the row-level CSV
    sim_first_col = []
    sim_prev_col = []
    for _, g in matches_df.groupby("track_id", sort=True):
        embs = np.stack(g["embedding"].to_numpy(), axis=0).astype(np.float32)
        ref = embs[0]
        for i, emb in enumerate(embs):
            sim_first_col.append(cosine_similarity(emb, ref))
            if i == 0:
                sim_prev_col.append(np.nan)
            else:
                sim_prev_col.append(cosine_similarity(emb, embs[i - 1]))
    matches_df["cosine_to_first"] = np.array(sim_first_col, dtype=np.float32)
    matches_df["cosine_to_prev"] = np.array(sim_prev_col, dtype=np.float32)

    # Save matches CSV without raw embedding vector column (too long), and with separate NPZ per track.
    matches_csv = matches_df.drop(columns=["embedding"]).copy()
    matches_csv.to_csv(args.output_dir / "track_matches.csv", index=False)

    summary_rows = []
    all_sim_to_first: List[float] = []

    eligible_groups = []
    for track_id, group in matches_df.groupby("track_id", sort=True):
        if len(group) < args.min_track_len:
            continue
        eligible_groups.append((track_id, group))

    for track_id, group in eligible_groups:
        summary = summarize_track(group)
        summary_rows.append({k: v for k, v in summary.items() if k not in {"embeddings", "sim_to_first", "consecutive_sim", "pairwise_matrix"}})

        track_dir = args.output_dir / "per_track"
        np.savez_compressed(
            track_dir / f"track_{int(track_id):06d}.npz",
            track_id=np.int32(track_id),
            frame_idx=group["frame_idx"].to_numpy(dtype=np.int32),
            embeddings=summary["embeddings"].astype(np.float32),
            sim_to_first=summary["sim_to_first"].astype(np.float32),
            consecutive_sim=summary["consecutive_sim"].astype(np.float32),
            pairwise_cosine=summary["pairwise_matrix"].astype(np.float32),
            iou=group["iou"].to_numpy(dtype=np.float32),
            score=group["score"].to_numpy(dtype=np.float32),
            label=group["label"].to_numpy(dtype=np.int32),
        )

        per_track_csv = group.drop(columns=["embedding"]).copy()
        per_track_csv.to_csv(track_dir / f"track_{int(track_id):06d}_matches.csv", index=False)

        all_sim_to_first.extend(summary["sim_to_first"].tolist())

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_dir / "track_summary.csv", index=False)

    # Global histogram
    if all_sim_to_first:
        plt.figure(figsize=(8, 5))
        plt.hist(all_sim_to_first, bins=30)
        plt.title("Global cosine similarity to first embedding")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(args.output_dir / "plots" / "global_hist_similarity_to_first.png", dpi=160)
        plt.close()

    # Boxplot by track id
    if summary_rows:
        top_track_ids = summary_df.sort_values("track_length", ascending=False)["track_id"].head(args.top_n_plots).astype(int).tolist()
        data = []
        labels = []
        for tid in top_track_ids:
            g = matches_df[matches_df["track_id"] == tid].sort_values("frame_idx")
            embs = np.stack(g["embedding"].to_numpy(), axis=0).astype(np.float32)
            ref = embs[0]
            sims = [cosine_similarity(e, ref) for e in embs]
            data.append(sims)
            labels.append(str(tid))

        if data:
            plt.figure(figsize=(max(8, len(data) * 0.45), 5))
            plt.boxplot(data, labels=labels, showfliers=False)
            plt.title("Cosine similarity to first embedding by track")
            plt.xlabel("Track ID")
            plt.ylabel("Cosine similarity")
            plt.tight_layout()
            plt.savefig(args.output_dir / "plots" / "boxplot_similarity_by_track.png", dpi=160)
            plt.close()

    # Per-track plots for top-N longest tracks
    for tid in summary_df.sort_values("track_length", ascending=False)["track_id"].head(args.top_n_plots).astype(int).tolist() if not summary_df.empty else []:
        g = matches_df[matches_df["track_id"] == tid].sort_values("frame_idx")
        embs = np.stack(g["embedding"].to_numpy(), axis=0).astype(np.float32)
        ref = embs[0]
        sim_to_first = np.array([cosine_similarity(e, ref) for e in embs], dtype=np.float32)
        frames = g["frame_idx"].to_numpy(dtype=np.int32)

        # Histogram per track
        plt.figure(figsize=(7, 4))
        plt.hist(sim_to_first, bins=20)
        plt.title(f"Track {tid} - histogram of cosine similarity to first")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(args.output_dir / "plots" / f"track_{tid:06d}_hist.png", dpi=160)
        plt.close()

        # Timeline plot
        plt.figure(figsize=(8, 4))
        plt.plot(frames, sim_to_first, marker="o", linewidth=1)
        plt.title(f"Track {tid} - cosine similarity to first over time")
        plt.xlabel("Frame index")
        plt.ylabel("Cosine similarity")
        plt.tight_layout()
        plt.savefig(args.output_dir / "plots" / f"track_{tid:06d}_timeline.png", dpi=160)
        plt.close()

        # Pairwise heatmap
        pairwise = pairwise_cosine_matrix(embs)
        plt.figure(figsize=(6, 5))
        plt.imshow(pairwise, vmin=-1.0, vmax=1.0, cmap="viridis")
        plt.colorbar(label="Cosine similarity")
        plt.title(f"Track {tid} - pairwise cosine similarity")
        plt.xlabel("Occurrence index")
        plt.ylabel("Occurrence index")
        plt.tight_layout()
        plt.savefig(args.output_dir / "plots" / f"track_{tid:06d}_pairwise_heatmap.png", dpi=160)
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
        "num_tracks_in_summary": int(len(summary_df)),
    }
    with (args.output_dir / "analysis_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report.keys()))
        writer.writeheader()
        writer.writerow(report)

    print(f"Analysis done. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract DEIMv2 detections + per-detection embeddings from a video.

Example:
python extract_video_detections_embeddings.py \
    --video path/to/video.mp4 \
    --config configs/deimv2/deimv2_dinov3_s_coco.yml \
    --ckpt checkpoints/deimv2_dinov3_s_coco.pth \
    --output_dir outputs/video_embeddings \
    --conf 0.40 \
    --device cuda \
    --image_size 640 640

Embedding provenance:
- `embeddings` comes from DEIM decoder query embeddings at the final decoder layer
  (`output` in `TransformerDecoder.forward`), propagated as `pred_embeddings`
  and then gathered in `PostProcessor` with the same top-k indices used for detections.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))
from engine.core import YAMLConfig
from engine.data.dataset import mscoco_category2name


DEFAULT_VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract per-frame DEIMv2 detections and embeddings from video")
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument("--config", type=Path, default=Path("configs/deimv2/deimv2_dinov3_s_coco.yml"), help="Model config")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path (.pth)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--image_size", type=int, nargs=2, metavar=("H", "W"), default=[640, 640], help="Inference resize")
    parser.add_argument("--vehicle_classes", nargs="+", default=DEFAULT_VEHICLE_CLASSES,
                        help="Class names to keep (default: car bus truck motorcycle)")
    parser.add_argument("--progress_every", type=int, default=50, help="Print progress every N frames")
    parser.add_argument("--save_debug_video", action="store_true", help="Save optional annotated debug video")
    parser.add_argument("--debug_video_name", type=str, default="debug_detections.mp4")
    return parser.parse_args()


class InferenceModel(nn.Module):
    def __init__(self, cfg: YAMLConfig):
        super().__init__()
        self.model = cfg.model.eval()
        self.postprocessor = cfg.postprocessor.eval()

    def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


def load_model(config_path: Path, ckpt_path: Path, device: torch.device) -> InferenceModel:
    cfg = YAMLConfig(str(config_path), resume=str(ckpt_path))

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

    cfg.model.load_state_dict(state)
    model = InferenceModel(cfg).to(device)
    model.eval()
    return model


def build_transform(image_size: Sequence[int], vit_backbone: bool):
    transforms = [T.Resize(tuple(image_size)), T.ToTensor()]
    if vit_backbone:
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def annotate_frame(frame_bgr: np.ndarray, boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray, label_names: List[str]) -> np.ndarray:
    out = frame_bgr.copy()
    for box, label, score, name in zip(boxes, labels, scores, label_names):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{name}({int(label)}): {score:.2f}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    cfg = YAMLConfig(str(args.config), resume=str(args.ckpt))
    vit_backbone = bool(cfg.yaml_cfg.get("DINOv3STAs", False))
    transform = build_transform(args.image_size, vit_backbone)

    model = load_model(args.config, args.ckpt, device)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    debug_writer = None
    if args.save_debug_video:
        debug_path = args.output_dir / args.debug_video_name
        debug_writer = cv2.VideoWriter(
            str(debug_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 25.0,
            (src_w, src_h),
        )

    vehicle_names = {name.strip().lower() for name in args.vehicle_classes}
    rows_for_index: List[Dict[str, object]] = []

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1

        orig_h, orig_w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        im_tensor = transform(frame_pil).unsqueeze(0).to(device)
        orig_target_sizes = torch.tensor([[orig_w, orig_h]], dtype=torch.float32, device=device)

        with torch.no_grad():
            outputs = model(im_tensor, orig_target_sizes)

        if not outputs:
            raise RuntimeError("Postprocessor returned empty batch output list.")

        det = outputs[0]
        if "embeddings" not in det:
            raise RuntimeError(
                "Model output does not include `embeddings`. "
                "Ensure DEIM decoder and postprocessor propagate final query embeddings."
            )

        labels = to_numpy(det["labels"]).astype(np.int32, copy=False)
        boxes = to_numpy(det["boxes"]).astype(np.float32, copy=False)
        scores = to_numpy(det["scores"]).astype(np.float32, copy=False)
        embeddings = to_numpy(det["embeddings"]).astype(np.float32, copy=False)

        label_names = np.array([mscoco_category2name.get(int(l), f"class_{int(l)}") for l in labels], dtype=object)

        keep_vehicle = np.array([name.lower() in vehicle_names for name in label_names], dtype=bool)
        keep_conf = scores >= float(args.conf)
        keep = keep_vehicle & keep_conf

        labels = labels[keep]
        boxes = boxes[keep]
        scores = scores[keep]
        embeddings = embeddings[keep]
        label_names = label_names[keep]

        if embeddings.size == 0:
            emb_l2 = embeddings.reshape(0, embeddings.shape[-1] if embeddings.ndim == 2 else 0)
        else:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            emb_l2 = embeddings / np.clip(norms, 1e-12, None)

        detection_ids = np.arange(labels.shape[0], dtype=np.int32)

        frame_file = args.output_dir / f"frame_{frame_idx:06d}.npz"
        np.savez_compressed(
            frame_file,
            frame_idx=np.int32(frame_idx),
            original_height=np.int32(orig_h),
            original_width=np.int32(orig_w),
            model_input_height=np.int32(args.image_size[0]),
            model_input_width=np.int32(args.image_size[1]),
            boxes_xyxy_abs=boxes.astype(np.float32),
            scores=scores.astype(np.float32),
            labels=labels.astype(np.int32),
            label_names=label_names,
            embeddings=embeddings.astype(np.float32),
            embeddings_l2norm=emb_l2.astype(np.float32),
            detection_ids=detection_ids,
        )

        rows_for_index.append({
            "frame_idx": frame_idx,
            "npz_file": frame_file.name,
            "num_detections": int(labels.shape[0]),
            "original_width": orig_w,
            "original_height": orig_h,
            "model_input_width": int(args.image_size[1]),
            "model_input_height": int(args.image_size[0]),
        })

        if debug_writer is not None:
            debug_frame = annotate_frame(frame_bgr, boxes, labels, scores, label_names.tolist())
            debug_writer.write(debug_frame)

        if frame_idx % max(args.progress_every, 1) == 0:
            tail = f"/{total_frames}" if total_frames > 0 else ""
            print(f"Processed frame {frame_idx}{tail}")

    cap.release()
    if debug_writer is not None:
        debug_writer.release()

    index_csv = args.output_dir / "frames_index.csv"
    with index_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_idx", "npz_file", "num_detections",
            "original_width", "original_height", "model_input_width", "model_input_height"
        ])
        writer.writeheader()
        writer.writerows(rows_for_index)

    index_json = args.output_dir / "run_metadata.json"
    metadata = {
        "video": str(args.video),
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "device": str(device),
        "confidence_threshold": float(args.conf),
        "vehicle_classes": sorted(vehicle_names),
        "image_size": [int(args.image_size[0]), int(args.image_size[1])],
        "fps": float(fps) if fps > 0 else None,
        "source_width": src_w,
        "source_height": src_h,
        "num_frames_processed": frame_idx,
    }
    index_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Done. Saved {frame_idx} frame files to: {args.output_dir}")


if __name__ == "__main__":
    main()

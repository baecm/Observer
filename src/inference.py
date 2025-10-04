#!/usr/bin/env python
# src/inference.py

import os
import gc
import argparse
import json
import torch
from torch.utils.data import DataLoader, Subset
import tqdm
import multiprocessing

from dataset.inference_dataset import InferenceDataset
from model.maskrcnn_builder import get_model_instance_segmentation

import config

from logger import Logger


def collate_fn(batch):
    """Bundle a batch into ([images], [metadata]) where metadata is [(replay_id, frame_id), ...]."""
    images, metas = zip(*batch)
    return list(images), list(metas)


def _available_cpu_count() -> int:
    """Estimate usable CPU cores (affinity-aware if possible)."""
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return multiprocessing.cpu_count()


def _auto_num_workers(device: torch.device) -> int:
    """
    Heuristic for DataLoader workers:
      - GPU: max(1, avail-1) to keep I/O busy without oversubscription
      - CPU: max(0, avail-1) to avoid contention with compute
    """
    avail = _available_cpu_count()
    if device.type == "cuda":
        return max(1, avail - 1)
    return max(0, avail - 1)


def _load_model(model_path: str, device: torch.device, in_channels: int, window_size: int, num_classes: int = 2):
    """Build the model, load weights (non-strict), move to device, and set eval mode."""
    model = get_model_instance_segmentation(
        num_classes=num_classes,
        in_channels=in_channels,
        window_size=window_size,
    )
    state = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0:
        Logger.warn(f"[Inference] Missing keys: {missing}")
    if len(unexpected) > 0:
        Logger.warn(f"[Inference] Unexpected keys: {unexpected}")
    model.to(device)
    model.eval()
    return model


def run_inference(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.5,
):
    """
    Run inference on a dataset and collect per-frame predictions.
    Note: masks are omitted to keep memory usage low.
    """
    model.eval()
    replay_results = []

    with torch.inference_mode():
        for images, metas in tqdm.tqdm(data_loader, desc="Running inference for replay", unit="batch"):
            # images: list of tensors [C,H,W]; metas: list of (rid, frame_id)
            images = [img.to(device, non_blocking=True) for img in images]
            outputs = model(images)  # list[dict], length = batch_size

            for output, (rid, frame_id) in zip(outputs, metas):
                # Filter by score threshold
                scores_all = output["scores"].detach().cpu().numpy().tolist()
                keep_idx = [i for i, s in enumerate(scores_all) if s >= score_threshold]

                boxes_all = output["boxes"].detach().cpu().numpy().tolist()
                labels_all = output["labels"].detach().cpu().numpy().tolist()

                frame_boxes, frame_scores, frame_labels = [], [], []
                for idx in keep_idx:
                    frame_boxes.append(boxes_all[idx])
                    frame_scores.append(scores_all[idx])
                    frame_labels.append(labels_all[idx])

                entry = {
                    "frame_id": frame_id,
                    "boxes": frame_boxes,
                    "scores": frame_scores,
                    "labels": frame_labels,
                }
                replay_results.append(entry)

            # free per-batch refs
            del outputs, images

    return replay_results


def save_predictions_as_coco(
    replay_id: str,
    replay_results: list,
    label_method: str,
    output_dir: str,
):
    """
    Save predictions for one replay in a COCO-style JSON (bbox-only; masks omitted).

    Output path mirrors GT structure:
        <output_dir>/<replay_id>.rep/<label_method>.json
    """
    out_dir = os.path.join(output_dir, f"{replay_id}.rep")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label_method}.json")

    categories = [{"id": 1, "name": "viewport", "supercategory": "viewport"}]

    coco = {
        "info": {
            "description": f"Predictions for replay {replay_id}",
            "version": "1.0",
            "label_method": label_method,
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    seen_frames = set()
    ann_id = 1

    for item in replay_results:
        fid = int(item["frame_id"])

        # One image entry per frame
        if fid not in seen_frames:
            coco["images"].append(
                {
                    "id": fid,
                    "file_name": f"{replay_id}.rep/{fid}.npy",
                    "width": int(config.ORIGIN_SHAPE[1]),
                    "height": int(config.ORIGIN_SHAPE[0]),
                }
            )
            seen_frames.add(fid)

        # Annotations (bbox-only; segmentation polygon derived from bbox)
        for box, score, label in zip(item["boxes"], item["scores"], item["labels"]):
            x1, y1, x2, y2 = map(int, box)
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            segmentation = [[x1, y1, x1 + w, y1, x1 + w, y1 + h, x1, y1 + h]]

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": fid,
                    "category_id": int(label),
                    "bbox": [x1, y1, w, h],
                    "score": float(score),
                    "area": int(w * h),
                    "segmentation": segmentation,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    Logger.info(f"[Inference] Saved predictions for replay {replay_id} -> {out_path}")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run Mask R-CNN inference on preprocessed StarCraft II replays")

    # Data and I/O
    group_data = parser.add_argument_group("Data and I/O")
    group_data.add_argument("--replays", nargs="+", required=True, help="Replay IDs to run inference on.")
    group_data.add_argument(
        "--data-root", type=str, default=os.path.join(os.getcwd(), "data"), help="Root directory for data."
    )
    group_data.add_argument(
        "--output-dir", type=str, default=os.path.join(os.getcwd(), "predictions"), help="Directory to save JSONs."
    )
    group_data.add_argument(
        "--include-components",
        type=str,
        nargs='+',
        default=['worker', 'ground', 'air', 'building', 'vision'],
        help="Components to include.",
    )
    group_data.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Subdirectory under --output-dir for predictions. "
            "If omitted, defaults to '<model_name>/model_<model_number>'."
        ),
    )

    # Model loading
    group_model = parser.add_argument_group("Model Loading")
    group_model.add_argument(
        "--model-root", type=str, default=os.path.join(os.getcwd(), "models"), help="Root directory for checkpoints."
    )
    group_model.add_argument("--model-name", type=str, required=True, help="Model folder name.")
    group_model.add_argument(
        "--model-number", type=int, required=True, help="Checkpoint number (e.g., 4 for model_004.pth)."
    )
    group_model.add_argument(
        "--label-method",
        type=str,
        default=config.LABEL_METHODS[0],
        choices=config.LABEL_METHODS,
        help="Label method for reference (not used during inference).",
    )
    group_model.add_argument(
        "--window-size", type=int, default=1, help="Window size used during training (must match)."
    )

    # Inference hyperparameters
    group_hyper = parser.add_argument_group("Inference Hyperparameters")
    group_hyper.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    group_hyper.add_argument(
        "--score-threshold", type=float, default=0.5, help="Score threshold for filtering predictions."
    )
    group_hyper.add_argument(
        "--sample-ratio", type=float, default=1.0, help="Fraction of frames to sample (0.0 < ratio ≤ 1.0)."
    )
    group_hyper.add_argument(
        "--workers", type=int, default=os.cpu_count() // 4, help="DataLoader workers. -1 = auto."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    Logger.info("[Inference] Starting…")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logger.info(f"[Inference] Using device: {device}")

    # Checkpoint path
    model_folder = os.path.join(args.model_root, args.model_name)
    model_path = os.path.join(model_folder, f"model_{args.model_number:03d}.pth")
    Logger.info(f"[Inference] Checkpoint path: {model_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Output directory for this run
    run_name = args.run_name or os.path.join(args.model_name, f"model_{args.model_number:03d}")
    run_dir = os.path.join(args.output_dir, run_name)
    Logger.info(f"[Inference] Output run dir: {run_dir}")

    # Infer input channels once from a small temporary dataset (first replay)
    temp_input_root = os.path.join(args.data_root, "input", "dst")
    temp_dataset = InferenceDataset(
        temp_input_root,
        [args.replays[0]],
        window_size=args.window_size,
        include_components=args.include_components,
    )
    in_channels = len(temp_dataset.channel_indices) * temp_dataset.window_size
    del temp_dataset

    # Build and load the model once
    model = _load_model(
        model_path=model_path,
        device=device,
        in_channels=in_channels,
        window_size=args.window_size,
        num_classes=2,
    )

    input_root = os.path.join(args.data_root, "input", "dst")
    for replay_id in args.replays:
        Logger.info(f"--- Processing replay: {replay_id} ---")

        # Dataset for this replay
        dataset = InferenceDataset(
            input_root,
            [replay_id],
            window_size=args.window_size,
            include_components=args.include_components,
        )

        if 0.0 < args.sample_ratio < 1.0:
            total_len = len(dataset)
            sample_size = int(total_len * args.sample_ratio)
            indices = torch.randperm(total_len).tolist()[:sample_size]
            dataset = Subset(dataset, indices)
            Logger.info(
                f"[Inference] Applied sampling: {sample_size}/{total_len} frames for replay {replay_id}"
            )

        # DataLoader config
        if args.workers is not None and args.workers >= 0:
            num_workers = args.workers
        else:
            num_workers = _auto_num_workers(device)

        Logger.info(
            f"[Inference] Dataset frames for {replay_id}: {len(dataset)}; num_workers={num_workers}"
        )

        dl_kwargs = dict(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,          # safer for long runs
            persistent_workers=False,  # ensure cleanup per replay
        )
        data_loader = DataLoader(dataset, **dl_kwargs)

        # Run inference
        replay_results = run_inference(
            model=model,
            data_loader=data_loader,
            device=device,
            score_threshold=args.score_threshold,
        )

        # Save predictions (COCO-style, bbox-only)
        save_predictions_as_coco(
            replay_id=replay_id,
            replay_results=replay_results,
            label_method=args.label_method,
            output_dir=run_dir,
        )

        # Cleanup per replay
        del data_loader, dataset, replay_results
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Final cleanup
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    Logger.info("[Inference] Complete!")


if __name__ == "__main__":
    main()

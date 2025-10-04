# src/train.py
import os
import argparse
import time
import tqdm
import utils
import json
import pickle
from multiprocessing import Pool

import torch
from torch.utils.data import Subset

import detection.transforms as T
from detection.engine import train_one_epoch, evaluate
from dataset.custom_penn_fudan import CustomPennFudanDataset
from model.maskrcnn_builder import get_model_instance_segmentation

import config
from .logger import Logger


def get_transform(train: bool):
    """Build a basic transform pipeline: ToTensor (+ RandomHorizontalFlip for training)."""
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int):
    """Create a DataLoader with the project-specific collate function."""
    if ds is None:
        return None
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=utils.collate_fn,
    )


def _process_json_worker(args):
    """Worker function to convert a single COCO JSON to a pickled subset for faster loading."""
    rid, label_root, label_method = args
    json_path = os.path.join(label_root, f"{rid}.rep", f"{label_method}.json")
    pkl_path = os.path.join(label_root, f"{rid}.rep", f"{label_method}.pkl")

    if not os.path.exists(json_path):
        return f"Skipped {rid}: no JSON found."
    if os.path.exists(pkl_path):
        return f"Skipped {rid}: pickle already exists."

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # Ensure required top-level keys exist.
        coco.setdefault("info", {"description": "auto-generated", "version": "1.0"})
        coco.setdefault("licenses", [])
        coco.setdefault("categories", [{"id": 1, "name": "viewport"}])
        coco.setdefault("images", [])
        coco.setdefault("annotations", [])

        with open(pkl_path, "wb") as f:
            pickle.dump({
                "info": coco["info"],
                "licenses": coco["licenses"],
                "categories": coco["categories"],
                "images": coco["images"],
                "annotations": coco["annotations"],
            }, f)
        return f"Success {rid}: pickle created."
    except Exception as e:
        return f"Failed {rid}: {e}"


def preprocess_json_to_pickle(label_root: str, label_method: str, replay_ids, num_workers: int, verbose: bool = True):
    """Convert COCO JSON files to pickled subsets in parallel for a list of replays."""
    def log(msg: str):
        if verbose:
            Logger.info(f"[Preprocess] {msg}")

    replay_ids = [str(r) for r in replay_ids]
    log(f"Starting JSON→Pickle conversion for {len(replay_ids)} replays using {num_workers} workers.")

    tasks = [(rid, label_root, label_method) for rid in replay_ids]

    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap_unordered(_process_json_worker, tasks),
                total=len(tasks),
                desc="Preprocessing JSON to Pickle",
            )
        )

    success_count = sum(1 for r in results if r.startswith("Success"))
    skipped_exist_count = sum(1 for r in results if "pickle already exists" in r)
    skipped_no_json_count = sum(1 for r in results if "no JSON found" in r)
    failed_count = sum(1 for r in results if r.startswith("Failed"))

    log(
        f"Preprocessing complete. Success: {success_count}, "
        f"Skipped (existing): {skipped_exist_count}, "
        f"Skipped (no JSON): {skipped_no_json_count}, "
        f"Failed: {failed_count}"
    )

    if failed_count > 0:
        for r in results:
            if r.startswith("Failed"):
                log(r)


def load_data(
    input_root: str,
    label_root: str,
    label_method: str,
    window_size: int,
    interval: int,
    batch_size: int,
    num_workers: int,
    replays,
    sample_ratio: float = 1.0,
    include_components=None,
    val_count: int = 1000,
):
    """Load training (and optional validation) datasets and return their loaders."""
    Logger.info("[Stage] Loading data…")
    Logger.info(f"[Info] Input root: {input_root}")
    Logger.info(f"[Info] Label root: {label_root}, method: {label_method}")

    # Required: a non-empty list of replay IDs.
    if not replays:
        raise ValueError("Please provide at least one replay via --replays.")
    train_ids = [str(r) for r in replays]
    Logger.info(f"[Info] Train IDs: {train_ids}")

    # Build dataset (train only; no dedicated test split here).
    train_dataset = CustomPennFudanDataset(
        input_root,
        label_root,
        label_method,
        training_ids=train_ids,
        training=True,
        window_size=window_size,
        interval=interval,
        include_components=include_components,
    )
    Logger.info(f"[Info] Full dataset size: Train {len(train_dataset)}")
    Logger.info(f"[Info] Window size: {window_size}, Interval: {interval}")

    # ---- Train/Validation split (based on the original dataset) ----
    n_train = len(train_dataset)
    val_dataset = None
    if val_count and n_train > val_count:
        full_idx = torch.randperm(n_train).tolist()
        val_idx = full_idx[-val_count:]
        train_idx = full_idx[:-val_count]

        # Important: build Subset instances from the original dataset.
        train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(train_dataset.dataset, val_idx)

    # ---- Optional subsampling of the training set ----
    if sample_ratio < 1.0:
        n_train = len(train_dataset)
        keep = torch.randperm(n_train).tolist()[: int(n_train * sample_ratio)]
        train_dataset = Subset(train_dataset, keep)
        Logger.info(
            f"[Info] Applied sampling to train data (ratio={sample_ratio}): Train {len(train_dataset)}"
        )

    train_loader = make_loader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = (
        make_loader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
        if val_dataset is not None
        else None
    )
    return train_loader, val_loader


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    data_loader_train,
    data_loader_validation,
    device: torch.device,
    num_epochs: int,
    save_dir: str,
):
    """Standard training loop with periodic evaluation and checkpointing."""
    Logger.info("[Stage] Starting training loop…")
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_stats = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()

        eval_stats = evaluate(model, data_loader_validation, device=device) if data_loader_validation is not None else None

        # Log key metrics to console.
        Logger.info(
            " | ".join(
                [
                    f"Epoch {epoch}",
                    f"Loss={train_stats.loss.global_avg:.4f}",
                    f"Cls={train_stats.loss_classifier.global_avg:.4f}",
                    f"BBox={train_stats.loss_box_reg.global_avg:.4f}",
                    f"Mask={train_stats.loss_mask.global_avg:.4f}",
                    f"Obj={train_stats.loss_objectness.global_avg:.4f}",
                    f"RPNBox={train_stats.loss_rpn_box_reg.global_avg:.4f}",
                ]
            )
        )

        if eval_stats is not None and hasattr(eval_stats, "coco_eval"):
            stat_names = [
                "AP",
                "AP50",
                "AP75",
                "APs",
                "APm",
                "APl",
                "AR1",
                "AR10",
                "AR100",
                "ARs",
                "ARm",
                "ARl",
            ]
            for iou_type, coco_eval in eval_stats.coco_eval.items():
                summary = ", ".join(
                    f"{name}={coco_eval.stats[i]:.3f}" for i, name in enumerate(stat_names)
                )
                Logger.info(f"[Eval:{iou_type}] {summary}")

        # Save checkpoints every 5 epochs and on the final epoch.
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            save_path = os.path.join(save_dir, f"model_{epoch + 1:03d}.pth")
            torch.save(model.state_dict(), save_path)
            Logger.info(f"[Info] Saved model checkpoint: {save_path}")


def run_training(args):
    Logger.info("[Stage] Preparing environment…")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    Logger.info(f"[Info] Using device: {device}")

    if not args.id_string:
        args.id_string = f"{args.label_method}_win{args.window_size}_b{args.batch_size}"

    log_save_path = os.path.join(args.log_root, f"{args.id_string}_{time.strftime('%Y%m%d_%H%M%S')}/")
    os.makedirs(log_save_path, exist_ok=True)
    Logger.info(f"[Info] Log save path: {log_save_path}")

    input_root = os.path.join(args.data_root, "input/dst")
    label_root = os.path.join(args.data_root, "label/dst")

    # Convert JSON to Pickle only for the specified replays.
    preprocess_json_to_pickle(
        label_root=label_root,
        label_method=args.label_method,
        replay_ids=args.replays,
        num_workers=args.num_workers,
    )
    Logger.info("[Info] JSON→Pickle conversion completed.")

    data_loader_train, data_loader_validation = load_data(
        input_root,
        label_root,
        args.label_method,
        args.window_size,
        args.interval,
        args.batch_size,
        args.num_workers,
        replays=args.replays,
        sample_ratio=args.sample_ratio,
        include_components=args.include_components,
        val_count=args.val_count,
    )

    if data_loader_validation is not None:
        Logger.info(
            f"[Info] Data loaded: Train {len(data_loader_train.dataset)}, Validation {len(data_loader_validation.dataset)}"
        )
    else:
        Logger.info(f"[Info] Data loaded: Train {len(data_loader_train.dataset)}, Validation (none)")

    Logger.info("[Stage] Initializing model…")
    num_classes = 2  # background + viewport

    train_ds = data_loader_train.dataset
    inner_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    in_channels = len(inner_ds.channel_indices) * inner_ds.window_size
    Logger.info(f"[Info] Input channels: {in_channels} (window size: {inner_ds.window_size})")

    model = get_model_instance_segmentation(
        num_classes=num_classes,
        window_size=args.window_size,
        in_channels=in_channels,
        do_normalize=args.do_normalize,
        normalize_mean=args.normalize_mean,
        normalize_std=args.normalize_std,
        resize_mode=args.resize_mode,
        min_sizes=args.min_sizes,
        max_size=args.max_size,
    )
    Logger.info(
        f"[Info] Model initialized with {num_classes} classes and {in_channels} input channels."
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_model(
        model,
        optimizer,
        lr_scheduler,
        data_loader_train,
        data_loader_validation,
        device,
        args.max_epoch,
        log_save_path,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Minimal argument parser for Mask R-CNN training")

    # Data and labeling
    group_data = parser.add_argument_group("Data and Labeling")
    group_data.add_argument(
        "--replays", type=str, nargs="+", required=True, help="List of replay IDs to use (train set = whole set)."
    )
    group_data.add_argument(
        "--label-method",
        type=str,
        default=config.LABEL_METHODS[0],
        choices=config.LABEL_METHODS,
        help="Label extraction method (folder name).",
    )
    group_data.add_argument("--sample-ratio", type=float, default=1.0, help="Fraction of training dataset to sample.")
    group_data.add_argument(
        "--data-root", type=str, default=os.path.join(os.getcwd(), "data"), help="Root directory for data."
    )
    group_data.add_argument(
        "--include-components",
        type=str,
        nargs='+',
        default=['worker', 'ground', 'air', 'building', 'vision'],
        help="List of components to include.",
    )
    group_data.add_argument(
        "--interval", type=int, default=config.INTERVAL, help="Sampling interval for frame windows (1 = every index)."
    )
    group_data.add_argument(
        "--val-count", type=int, default=1000, help="Number of samples for validation (0 = no validation)."
    )

    # Model hyperparameters
    group_hyper = parser.add_argument_group("Model Hyperparameters")
    group_hyper.add_argument("--window-size", type=int, default=config.WINDOW_SIZE)
    group_hyper.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    group_hyper.add_argument("--learning-rate", type=float, default=config.TRAIN_LEARNING_RATE)
    group_hyper.add_argument("--max-epoch", type=int, default=config.TRAIN_EPOCHS)

    # Transform / resize / normalize
    group_tf = parser.add_argument_group("Transform / Resize / Normalize")
    group_tf.add_argument(
        "--resize-mode",
        type=str,
        choices=["resize", "keep"],
        default="resize",
        help="'resize' = scale to target sizes (recommended), 'keep' = preserve original size.",
    )
    group_tf.add_argument(
        "--min-sizes",
        type=int,
        nargs='+',
        default=[800],
        help="Multiscale example: 640 800 896 960 1024 (only used when resize-mode=resize)",
    )
    group_tf.add_argument("--max-size", type=int, default=1333)
    group_tf.add_argument("--do-normalize", action="store_true", help="Apply per-channel mean/std normalization.")
    group_tf.add_argument("--normalize-mean", type=float, nargs='+', help="Normalization means (length = in_channels)")
    group_tf.add_argument("--normalize-std", type=float, nargs='+', help="Normalization stds (length = in_channels)")

    # Environment and logging
    group_env = parser.add_argument_group("Environment and Logging")
    group_env.add_argument("--cuda", action='store_true', default=True, help="Enable CUDA training.")
    group_env.add_argument("--id-string", type=str, default="", help="Identifier string for the training run.")
    group_env.add_argument(
        "--log-level", type=str, default="log", choices=["none", "log", "debug"], help="Logging level."
    )
    group_env.add_argument(
        "--log-root", type=str, default=os.path.join(os.getcwd(), "models"), help="Root for models and logs."
    )
    group_env.add_argument(
        "--num-workers", type=int, default=os.cpu_count() // 4, help="Number of CPU workers for data loading."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    Logger.set_level(args.log_level)

    try:
        Logger.info("[Entry] Starting training script…")
        run_training(args)
    except Exception as e:
        if not args.id_string:
            args.id_string = f"{args.label_method}_win{args.window_size}_b{args.batch_size}"
        error_message = f"Training run '{args.id_string}' failed with an error: {e}"
        Logger.error(error_message)
        raise

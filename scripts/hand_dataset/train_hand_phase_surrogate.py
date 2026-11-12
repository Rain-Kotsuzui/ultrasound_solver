"""Train the additive-residual hand BEM response surrogate."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch.utils.data import DataLoader

from algorithm.training.hand_phase_dataset import (
    HandPhaseSurrogateDataset,
    load_array_metadata,
    load_dataset_manifest,
    move_batch_to_device,
)
from algorithm.training.hand_phase_surrogate import (
    AdditiveResidualBEMSurrogate,
    AdditiveResidualSurrogateConfig,
    additive_residual_surrogate_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default=str(ROOT / "outputs/hand_phase_surrogate_mvp_v1"),
        help="Dataset directory containing dataset.json, array_metadata.npz and samples/",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs/hand_phase_surrogate_train_v1"),
        help="Directory for checkpoints and metrics",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--target-subsample", type=int, default=0,
                        help="Targets per geometry per epoch item; 0 keeps all targets")
    parser.add_argument("--max-train-geometries", type=int, default=0)
    parser.add_argument("--max-val-geometries", type=int, default=0)
    parser.add_argument("--overfit-geometries", type=int, default=0,
                        help="Use first N train geometries as both train and validation")
    parser.add_argument("--frequency-hz", type=float, default=40000.0)
    parser.add_argument("--sound-speed", type=float, default=343.0)
    parser.add_argument("--geometry-latent-dim", type=int, default=256)
    parser.add_argument("--target-latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--target-fourier-bands", type=int, default=8)
    parser.add_argument("--target-category-dim", type=int, default=0,
                        help="Optional sampling-category embedding dim; 0 disables target_category input")
    parser.add_argument("--geometry-blocks", type=int, default=4)
    parser.add_argument("--target-blocks", type=int, default=3)
    parser.add_argument("--decoder-blocks", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--response-weight", type=float, default=1.0)
    parser.add_argument("--log-amplitude-weight", type=float, default=0.1)
    parser.add_argument("--phase-weight", type=float, default=0.05)
    parser.add_argument(
        "--complex-loss",
        choices=("mse", "nmse"),
        default="mse",
        help="Complex response loss. 'nmse' normalizes each batch item by BEM response energy.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", default="",
                        help="Resume from 'latest', 'best', or an explicit checkpoint path")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="Print in-epoch progress every N batches; 0 disables batch progress")
    parser.add_argument("--progress-seconds", type=float, default=10.0,
                        help="Minimum seconds between in-epoch progress prints")
    return parser.parse_args()


def _positive_or_none(value: int) -> int | None:
    return value if value and value > 0 else None


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(
    dataset_dir: Path,
    split: str,
    *,
    batch_size: int,
    max_geometries: int | None,
    target_subsample: int | None,
    seed: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = HandPhaseSurrogateDataset(
        dataset_dir,
        split,
        max_geometries=max_geometries,
        target_subsample=target_subsample,
        target_seed=seed,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def _metrics_from_prediction(prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    target = torch.complex(batch["response_real"], batch["response_imag"])
    error = prediction["response"] - target
    nmse = (error.abs().square().sum() / target.abs().square().sum().clamp_min(1.0e-12)).detach()
    amp_rel = (
        (prediction["response"].abs() - target.abs()).abs().mean()
        / target.abs().mean().clamp_min(1.0e-12)
    ).detach()
    phase_error = torch.angle(prediction["response"]) - torch.angle(target)
    phase_mae = torch.atan2(torch.sin(phase_error), torch.cos(phase_error)).abs().mean().detach()
    return {
        "complex_nmse": float(nmse.cpu()),
        "amplitude_relative_error": float(amp_rel.cpu()),
        "phase_mae_rad": float(phase_mae.cpu()),
    }


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _print_progress(
    *,
    phase: str,
    epoch: int,
    batch_index: int,
    batch_count: int,
    phase_started: float,
    loss_value: float,
    running_total: float,
) -> None:
    elapsed = time.perf_counter() - phase_started
    done = batch_index + 1
    percent = 100.0 * done / max(batch_count, 1)
    avg = running_total / max(done, 1)
    rate = done / max(elapsed, 1.0e-9)
    remaining = (batch_count - done) / max(rate, 1.0e-9)
    print(
        f"[{phase}] epoch {epoch} batch {done}/{batch_count} "
        f"({percent:5.1f}%) loss={loss_value:.4f} avg={avg:.4f} "
        f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(remaining)}",
        flush=True,
    )


def _print_metric_header() -> None:
    print(
        "epoch  time   train_total  val_total  val_nmse  val_phase  val_amp  best_val  ckpt",
        flush=True,
    )
    print(
        "-----  -----  -----------  ---------  --------  ---------  -------  --------  ----",
        flush=True,
    )


def _print_metric_row(record: dict, best_val: float, checkpoint_note: str) -> None:
    validation = record["validation"]
    train = record["train"]
    print(
        f"{record['epoch']:>5d}  "
        f"{_format_seconds(record['elapsed_s']):>5}  "
        f"{train.get('total', float('nan')):>11.4f}  "
        f"{validation.get('total', float('nan')):>9.4f}  "
        f"{validation.get('complex_nmse', float('nan')):>8.4f}  "
        f"{validation.get('phase_mae_rad', float('nan')):>9.4f}  "
        f"{validation.get('amplitude_relative_error', float('nan')):>7.4f}  "
        f"{best_val:>8.4f}  "
        f"{checkpoint_note}",
        flush=True,
    )


def _append_metric_table(path: Path, record: dict, best_val: float, checkpoint_note: str) -> None:
    if not path.exists():
        path.write_text(
            "epoch\telapsed_s\tepoch_s\ttrain_total\tval_total\tval_complex_nmse\t"
            "val_phase_mae_rad\tval_amplitude_relative_error\tbest_val\tcheckpoint\n",
            encoding="utf-8",
        )
    train = record["train"]
    validation = record["validation"]
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"{record['epoch']}\t{record['elapsed_s']}\t{record['epoch_s']}\t"
            f"{train.get('total', float('nan'))}\t{validation.get('total', float('nan'))}\t"
            f"{validation.get('complex_nmse', float('nan'))}\t"
            f"{validation.get('phase_mae_rad', float('nan'))}\t"
            f"{validation.get('amplitude_relative_error', float('nan'))}\t"
            f"{best_val}\t{checkpoint_note}\n"
        )


def _best_val_from_metrics(path: Path) -> float:
    if not path.is_file():
        return float("inf")
    best = float("inf")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            value = float(record.get("validation", {}).get("total", float("inf")))
        except Exception:
            continue
        best = min(best, value)
    return best


def _run_epoch(
    model: AdditiveResidualBEMSurrogate,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    epoch: int,
    phase: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_items = 0
    accum: dict[str, float] = {}
    phase_started = time.perf_counter()
    last_progress_s = -float("inf")
    batch_count = len(loader)
    running_loss = 0.0
    for batch_index, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(training):
            prediction = model(
                shape_beta=batch["shape_beta"],
                hand_pose=batch["hand_pose"],
                root_rotation=batch["root_rotation"],
                root_translation_m=batch["root_translation_m"],
                target_world_m=batch["target_world_m"],
                target_category=batch["target_category"],
            )
            losses = additive_residual_surrogate_loss(
                prediction,
                batch["response_real"],
                batch["response_imag"],
                residual_weight=args.residual_weight,
                response_weight=args.response_weight,
                log_amplitude_weight=args.log_amplitude_weight,
                phase_weight=args.phase_weight,
                complex_loss=args.complex_loss,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        batch_size = int(batch["shape_beta"].shape[0])
        total_items += batch_size
        values = {key: float(value.detach().cpu()) for key, value in losses.items()}
        running_loss += values.get("total", 0.0)
        now = time.perf_counter()
        should_print_progress = (
            args.progress_every > 0
            and (
                (batch_index + 1) == batch_count
                or (batch_index + 1) % args.progress_every == 0
                or now - last_progress_s >= args.progress_seconds
            )
        )
        if should_print_progress:
            _print_progress(
                phase=phase,
                epoch=epoch,
                batch_index=batch_index,
                batch_count=batch_count,
                phase_started=phase_started,
                loss_value=values.get("total", float("nan")),
                running_total=running_loss,
            )
            last_progress_s = now
        values.update(_metrics_from_prediction(prediction, batch))
        for key, value in values.items():
            accum[key] = accum.get(key, 0.0) + value * batch_size
    return {key: value / max(total_items, 1) for key, value in accum.items()}


def _save_checkpoint(
    path: Path,
    *,
    model: AdditiveResidualBEMSurrogate,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    args: argparse.Namespace,
    config: AdditiveResidualSurrogateConfig,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
            "model_config": asdict(config),
            "wave_number": model.wave_number,
        },
        temporary,
    )
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return path
        except PermissionError:
            if attempt == 7:
                break
            time.sleep(0.25)

    fallback = path.with_name(f"{path.stem}_epoch_{epoch:04d}_{int(time.time())}{path.suffix}")
    os.replace(temporary, fallback)
    print(
        f"warning: could not replace locked checkpoint {path.name}; wrote {fallback.name} instead",
        flush=True,
    )
    return fallback


def _checkpoint_path(output_dir: Path, resume: str) -> Path:
    if resume in {"latest", "best"}:
        return output_dir / f"{resume}.pt"
    return Path(resume).expanduser().resolve()


def _load_checkpoint_if_requested(
    *,
    model: AdditiveResidualBEMSurrogate,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    resume: str,
    device: torch.device,
) -> tuple[int, dict | None]:
    if not resume:
        return 1, None
    path = _checkpoint_path(output_dir, resume)
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1
    return start_epoch, checkpoint


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")
    _seed_everything(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_dataset_manifest(dataset_dir)
    metadata = load_array_metadata(dataset_dir)
    centers = torch.as_tensor(metadata["centers_m"], dtype=torch.float32)
    amplitudes = torch.as_tensor(metadata.get("amplitudes", torch.ones(len(centers))), dtype=torch.float32)
    wave_number = 2.0 * math.pi * args.frequency_hz / args.sound_speed

    target_subsample = _positive_or_none(args.target_subsample)
    if args.overfit_geometries > 0:
        train_loader = _make_loader(
            dataset_dir,
            "train",
            batch_size=args.batch_size,
            max_geometries=args.overfit_geometries,
            target_subsample=target_subsample,
            seed=args.seed,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = _make_loader(
            dataset_dir,
            "train",
            batch_size=args.batch_size,
            max_geometries=args.overfit_geometries,
            target_subsample=target_subsample,
            seed=args.seed,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        train_loader = _make_loader(
            dataset_dir,
            "train",
            batch_size=args.batch_size,
            max_geometries=_positive_or_none(args.max_train_geometries),
            target_subsample=target_subsample,
            seed=args.seed,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = _make_loader(
            dataset_dir,
            "validation",
            batch_size=args.batch_size,
            max_geometries=_positive_or_none(args.max_val_geometries),
            target_subsample=None,
            seed=args.seed + 17,
            shuffle=False,
            num_workers=args.num_workers,
        )

    config = AdditiveResidualSurrogateConfig(
        num_emitters=int(centers.shape[0]),
        geometry_latent_dim=args.geometry_latent_dim,
        target_latent_dim=args.target_latent_dim,
        hidden_dim=args.hidden_dim,
        target_fourier_bands=args.target_fourier_bands,
        target_category_count=len(manifest.get("target_categories", {"uniform": 0, "near_hand": 1, "trajectory": 2})),
        target_category_dim=args.target_category_dim,
        geometry_blocks=args.geometry_blocks,
        target_blocks=args.target_blocks,
        decoder_blocks=args.decoder_blocks,
        dropout=args.dropout,
    )
    device = torch.device(args.device)
    model = AdditiveResidualBEMSurrogate(
        emitter_centers_m=centers,
        emitter_amplitudes=amplitudes,
        wave_number=wave_number,
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch, resumed_checkpoint = _load_checkpoint_if_requested(
        model=model,
        optimizer=optimizer,
        output_dir=output_dir,
        resume=args.resume,
        device=device,
    )
    if start_epoch > args.epochs:
        print(f"Checkpoint is already at epoch {start_epoch - 1}; requested --epochs {args.epochs}.")
        return

    run_config = {
        "args": vars(args),
        "model_config": asdict(config),
        "dataset_manifest": manifest,
        "train_geometries": len(train_loader.dataset),
        "validation_geometries": len(val_loader.dataset),
        "wave_number": wave_number,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    metrics_path = output_dir / "metrics.jsonl"
    metrics_table_path = output_dir / "metrics_table.tsv"
    best_val = _best_val_from_metrics(metrics_path)
    if resumed_checkpoint is not None:
        best_val = min(
            best_val,
            float(resumed_checkpoint.get("metrics", {}).get("validation", {}).get("total", float("inf"))),
        )
    started = time.perf_counter()
    print(json.dumps(run_config, indent=2))
    if resumed_checkpoint is not None:
        print(f"Resumed from epoch {start_epoch - 1}; continuing through epoch {args.epochs}.", flush=True)
    _print_metric_header()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = _run_epoch(
            model, train_loader, device=device, optimizer=optimizer, args=args, epoch=epoch, phase="train"
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model, val_loader, device=device, optimizer=None, args=args, epoch=epoch, phase="val"
            )
        record = {
            "epoch": epoch,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "epoch_s": round(time.perf_counter() - epoch_started, 3),
            "train": train_metrics,
            "validation": val_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

        latest_path = _save_checkpoint(
            output_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=record,
            args=args,
            config=config,
        )
        checkpoint_note = "latest" if latest_path.name == "latest.pt" else latest_path.stem
        val_total = val_metrics.get("total", float("inf"))
        if val_total < best_val:
            best_val = val_total
            best_path = _save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=record,
                args=args,
                config=config,
            )
            checkpoint_note += ",best" if best_path.name == "best.pt" else f",{best_path.stem}"
        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_path = _save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=record,
                args=args,
                config=config,
            )
            checkpoint_note += f",{epoch_path.stem}"
        _append_metric_table(metrics_table_path, record, best_val, checkpoint_note)
        if epoch != start_epoch and epoch % 20 == 0:
            _print_metric_header()
        _print_metric_row(record, best_val, checkpoint_note)


if __name__ == "__main__":
    main()

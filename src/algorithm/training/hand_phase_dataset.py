"""Dataset utilities for the MANO hand BEM response surrogate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - importable in non-ML envs
    torch = None
    Tensor = Any

    class Dataset:  # type: ignore[no-redef]
        pass


SAMPLE_KEYS = (
    "shape_beta",
    "hand_pose",
    "root_rotation",
    "root_translation_m",
    "target_world_m",
    "target_category",
    "response_real",
    "response_imag",
)


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("torch is required for hand phase surrogate datasets")


def find_sample_files(dataset_dir: str | Path, split: str) -> list[Path]:
    """Return sorted sample files for a split directory."""
    root = Path(dataset_dir)
    sample_dir = root / "samples" / split
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Missing sample split directory: {sample_dir}")
    files = sorted(sample_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz samples found in {sample_dir}")
    return files


def load_array_metadata(dataset_dir: str | Path) -> dict[str, np.ndarray]:
    """Load emitter centers and amplitudes saved by the dataset generator."""
    path = Path(dataset_dir) / "array_metadata.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing array metadata: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_dataset_manifest(dataset_dir: str | Path) -> dict:
    path = Path(dataset_dir) / "dataset.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


class HandPhaseSurrogateDataset(Dataset):
    """Geometry-level `.npz` dataset for BEM response surrogate training.

    Each item is one hand geometry with all target points kept together. The
    default generated MVP data therefore returns tensors shaped like
    ``target_world_m=(128, 3)`` and ``response_real=(128, 256)``.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        *,
        max_geometries: int | None = None,
        target_subsample: int | None = None,
        target_seed: int = 0,
    ) -> None:
        _require_torch()
        self.dataset_dir = Path(dataset_dir)
        self.split = str(split)
        self.files = find_sample_files(self.dataset_dir, self.split)
        if max_geometries is not None:
            if max_geometries < 1:
                raise ValueError("max_geometries must be positive")
            self.files = self.files[:max_geometries]
        self.target_subsample = target_subsample
        if target_subsample is not None and target_subsample < 1:
            raise ValueError("target_subsample must be positive")
        self.target_seed = int(target_seed)

    def __len__(self) -> int:
        return len(self.files)

    def _target_indices(self, count: int, item_index: int) -> np.ndarray | slice:
        if self.target_subsample is None or self.target_subsample >= count:
            return slice(None)
        rng = np.random.default_rng(self.target_seed + item_index * 1009)
        return np.sort(rng.choice(count, size=self.target_subsample, replace=False))

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in SAMPLE_KEYS if key not in data.files]
            if missing:
                raise KeyError(f"{path} is missing required keys: {missing}")
            target_count = int(data["target_world_m"].shape[0])
            target_indices = self._target_indices(target_count, index)
            sample = {
                "shape_beta": torch.from_numpy(np.asarray(data["shape_beta"], dtype=np.float32)),
                "hand_pose": torch.from_numpy(np.asarray(data["hand_pose"], dtype=np.float32)),
                "root_rotation": torch.from_numpy(np.asarray(data["root_rotation"], dtype=np.float32)),
                "root_translation_m": torch.from_numpy(np.asarray(data["root_translation_m"], dtype=np.float32)),
                "target_world_m": torch.from_numpy(np.asarray(data["target_world_m"][target_indices], dtype=np.float32)),
                "target_category": torch.from_numpy(np.asarray(data["target_category"][target_indices], dtype=np.int64)),
                "response_real": torch.from_numpy(np.asarray(data["response_real"][target_indices], dtype=np.float32)),
                "response_imag": torch.from_numpy(np.asarray(data["response_imag"][target_indices], dtype=np.float32)),
                "geometry_id": torch.as_tensor(int(data["geometry_id"]), dtype=torch.long),
            }
        return sample


def move_batch_to_device(batch: dict[str, Tensor], device: str | torch.device) -> dict[str, Tensor]:
    _require_torch()
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

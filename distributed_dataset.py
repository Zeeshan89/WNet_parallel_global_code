"""
distributed_dataset.py
─────────────────────────────────────────────────────────────────────────────
File-level sharding for true distributed training on Frontier.

KEY IDEA
────────
Instead of every GPU loading the same monthly file, we assign each GPU a
disjoint subset of monthly HDF5 files via round-robin.  DDP all_reduce
keeps model weights in sync after every gradient step.

ASSIGNMENT STRATEGY  (round-robin)
───────────────────
    rank 0  → files [0,  world_size,   2*world_size, ...]
    rank 1  → files [1,  world_size+1, 2*world_size+1, ...]
    rank k  → files [k,  world_size+k, 2*world_size+k, ...]

MEMORY BUDGET (Frontier MI300A / MI250X)
────────────────────────────────────────
  Each MI250X GCD has 64 GB HBM2e.
  Each monthly HDF5 file is ~3.3 GB on disk.
  After precomputing 14-channel tensors at (192,384) float32:
      ~124 samples × 14 × 192 × 384 × 4 bytes ≈ 640 MB per month
  → Well within budget even for 4–8 months loaded simultaneously.
"""

import os
from collections import defaultdict
import numpy as np
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Global-stratified train/val split (whole-month holdout)
# ─────────────────────────────────────────────────────────────────────────────

def _file_year_month(fname: str) -> Optional[Tuple[int, int]]:
    """Extract (year, month) from 'era5dara_YYYYMM.nc' style filenames."""
    try:
        date_str = fname.split("_")[1].split(".")[0]   # "198003"
        return int(date_str[:4]), int(date_str[4:6])
    except Exception:
        return None


def build_train_val_split(
    all_files: List[str],
    val_fraction: float = 0.135,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Global stratified train/val split by calendar month — whole-month holdout.

    For each calendar month (1..12), reserves ~val_fraction of that month's
    files as validation. Stratifying by month guarantees that every season
    (incl. hurricane season Aug-Oct) has held-out positives so per-class
    metrics are well-defined.

    Deterministic via per-month seed (seed + month), so adding/removing a
    calendar month doesn't shuffle the others' selections.

    Args:
        all_files     : Sorted list of monthly file names like 'era5dara_YYYYMM.nc'.
        val_fraction  : Target fraction held out per calendar month (~0.135 → ~13.5%).
        seed          : Base RNG seed.

    Returns:
        (train_files, val_files), each sorted, disjoint, union == all_files.
    """
    by_month = defaultdict(list)
    unparseable = []
    for f in all_files:
        ym = _file_year_month(f)
        if ym is None:
            unparseable.append(f)
            continue
        by_month[ym[1]].append(f)

    val_set = set()
    for month in sorted(by_month):
        files = sorted(by_month[month])
        n_val = max(1, int(round(len(files) * val_fraction)))
        n_val = min(n_val, len(files) - 1)  # leave at least 1 for training
        rng_m = np.random.default_rng(seed + month)
        chosen = rng_m.choice(len(files), size=n_val, replace=False)
        for i in chosen:
            val_set.add(files[i])

    train_files = sorted(f for f in all_files if f not in val_set)
    val_files   = sorted(val_set)
    return train_files, val_files


def log_train_val_split(
    rank: int,
    all_files: List[str],
    train_files: List[str],
    val_files: List[str],
) -> None:
    """Rank-0 summary of the global train/val split, with per-month breakdown."""
    if rank != 0:
        return
    n_all, n_tr, n_va = len(all_files), len(train_files), len(val_files)
    print(
        f"\n[Train/Val split] {n_all} total → "
        f"{n_tr} train ({100*n_tr/max(1,n_all):.1f}%) | "
        f"{n_va} val ({100*n_va/max(1,n_all):.1f}%)"
    )
    per_month_val = defaultdict(int)
    per_month_total = defaultdict(int)
    for f in all_files:
        ym = _file_year_month(f)
        if ym is None:
            continue
        per_month_total[ym[1]] += 1
        if f in set(val_files):
            per_month_val[ym[1]] += 1
    print("  Per-calendar-month val counts (mm: val/total):")
    line = "  "
    for m in range(1, 13):
        line += f" {m:02d}:{per_month_val[m]}/{per_month_total[m]}"
    print(line)
    if val_files:
        sample = ", ".join(val_files[:6]) + ("..." if len(val_files) > 6 else "")
        print(f"  Held-out months sample: {sample}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  File-level sharding helper
# ─────────────────────────────────────────────────────────────────────────────

def get_rank_files(
    all_files: List[str],
    rank: int,
    world_size: int,
    shuffle: bool = False,
    seed: int = 42,
) -> List[str]:
    """
    Return the subset of *all_files* that belongs to this rank.

    Round-robin assignment ensures even distribution and temporal diversity.

    Args:
        all_files  : Sorted list of ALL monthly file names.
        rank       : This process's global rank (0 … world_size-1).
        world_size : Total number of processes.
        shuffle    : Shuffle files before assignment (reproducible via seed).
        seed       : RNG seed used when shuffle=True.

    Returns:
        List of file names assigned to this rank.

    Example
    -------
    >>> files = ['f00.h5', 'f01.h5', 'f02.h5', 'f03.h5', 'f04.h5']
    >>> get_rank_files(files, rank=0, world_size=2)
    ['f00.h5', 'f02.h5', 'f04.h5']
    >>> get_rank_files(files, rank=1, world_size=2)
    ['f01.h5', 'f03.h5']
    """
    if world_size <= 1:
        return list(all_files)

    files = list(all_files)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(files)

    # Round-robin: rank gets indices rank, rank+world_size, rank+2*world_size, …
    return files[rank::world_size]


def log_file_assignment(
    rank: int,
    world_size: int,
    my_files: List[str],
    total_files: int,
) -> None:
    """Print a human-readable summary of the file assignment for this rank."""
    mine     = len(my_files)
    data_gb  = mine * 3.3   # approximate per-file size on disk

    print(
        f"\n[Rank {rank}/{world_size}] File-sharding summary\n"
        f"  Total files in range : {total_files}\n"
        f"  Files assigned to me : {mine}  (~{data_gb:.1f} GB on disk)\n"
        f"  My files             : {my_files[:4]}{'...' if mine > 4 else ''}\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Epoch-level file rotation scheduler
# ─────────────────────────────────────────────────────────────────────────────

class FileRotationScheduler:
    """
    Manages which monthly file each rank trains on during each epoch.

    Instead of loading ALL months at once (memory-intensive), this scheduler
    rotates through the assigned files epoch by epoch.  Each epoch, every
    rank trains on ONE of its assigned files.  After cycling through all
    assigned files it starts over (optionally reshuffling).

    Recommended for Frontier where you want:
      - Low memory usage (one month per GPU at a time)
      - Each GPU sees different data than other GPUs
      - All assigned months visited over training

    Usage
    -----
        scheduler = FileRotationScheduler(my_files, rank, seed=42)
        for global_epoch in range(total_epochs):
            current_file = scheduler.current_file()
            # ... build dataset and train on current_file ...
            scheduler.step()
    """

    def __init__(
        self,
        rank_files: List[str],
        rank: int,
        seed: int = 42,
        reshuffle_each_cycle: bool = True,
    ):
        self.rank              = rank
        self.seed              = seed
        self.reshuffle         = reshuffle_each_cycle
        self._original_files   = list(rank_files)
        self._files            = list(rank_files)
        self._idx              = 0
        self._cycle            = 0

    def current_file(self) -> Optional[str]:
        """Return the file for the current step."""
        if not self._files:
            return None
        return self._files[self._idx % len(self._files)]

    def step(self) -> None:
        """Advance to the next file; reshuffle at the start of each new cycle."""
        if not self._files:
            return
        self._idx += 1
        if self._idx >= len(self._files):
            self._idx   = 0
            self._cycle += 1
            if self.reshuffle:
                rng = np.random.default_rng(self.seed + self._cycle)
                rng.shuffle(self._files)
                if self.rank == 0:
                    print(f"[FileRotationScheduler] Starting cycle {self._cycle} "
                          f"(reshuffled {len(self._files)} files)")

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def position(self) -> int:
        return self._idx

    def __len__(self) -> int:
        return len(self._files)

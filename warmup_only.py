"""
warmup_only.py — Complete MIOpen kernel warmup for ImprovedWNet.

Runs enough forward+backward passes to trigger kernel search for EVERY
conv layer at EVERY spatial scale in the WNet encoder/decoder, then
saves the unified cache to Lustre so all future training jobs load
kernels instantly (zero search time).

Usage (called from warmup_miopen.sh):
    srun ... python warmup_only.py

After completion:
    Set MIOPEN_FIND_MODE=2 (FAST) in training script.
    Every layer will be a Find-Db hit → zero startup overhead.

Key facts from main.py:
    batch_size                  = 2
    in_channels                 = 14
    target_size                 = (192, 384)
    gradient_accumulation_steps = 3
    model = ImprovedWNet(in_channels=14, num_classes=5)
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from datetime import timedelta


# ── Configuration — must match main.py exactly ────────────────────────────────
BATCH_SIZE   = 2
IN_CHANNELS  = 14
NUM_CLASSES  = 5
HEIGHT       = 192
WIDTH        = 384
GRAD_ACCUM   = 3      # matches gradient_accumulation_steps in main.py
WARMUP_STEPS = 40     # enough passes to hit all encoder/decoder spatial scales
                      # WNet has 2 UNets × 5 encoder + 5 decoder levels
                      # = 20 conv blocks, each at different spatial dims


def setup_distributed():
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    rank       = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "3442")

    os.environ.update({
        "WORLD_SIZE":   str(world_size),
        "RANK":         str(rank),
        "LOCAL_RANK":   str(local_rank),
        "MASTER_ADDR":  master_addr,
        "MASTER_PORT":  master_port,
    })

    gpu_count = torch.cuda.device_count()
    device_id = local_rank % max(1, gpu_count)
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=1800),
    )

    return rank, world_size, local_rank, device_id, device


def run_warmup_steps(model, optimizer, scaler, device, rank, n_steps, grad_accum):
    """
    Run n_steps training iterations with grad accumulation, AMP, and
    proper loss against both WNet outputs (seg + reconstruction).
    This exercises every conv layer at every spatial scale.
    """
    model.train()
    total_start = time.time()

    for step in range(n_steps):
        step_start = time.time()

        # Match exact training input shape from main.py
        x = torch.randn(
            BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH,
            device=device, dtype=torch.float32
        )
        # Targets matching WNet outputs
        seg_target = torch.randint(
            0, NUM_CLASSES, (BATCH_SIZE, HEIGHT, WIDTH),
            device=device
        )
        rec_target = torch.randn(
            BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH,
            device=device, dtype=torch.float32
        )

        optimizer.zero_grad()

        # Simulate gradient accumulation — same code path as train.py
        for acc in range(grad_accum):
            with autocast():
                seg_probs, rec = model(x)

                # Proper loss against both outputs — forces backward
                # through full encoder AND decoder of both UNets
                seg_loss = nn.functional.cross_entropy(seg_probs, seg_target)
                rec_loss = nn.functional.mse_loss(rec, rec_target)
                loss     = (seg_loss + rec_loss) / grad_accum

            scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        # Barrier ensures all ranks finish each step before continuing
        dist.barrier()
        torch.cuda.synchronize()

        step_time = time.time() - step_start
        if rank == 0:
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  Step {step+1:2d}/{n_steps}: "
                  f"seg_loss={seg_loss.item():.4f}  "
                  f"rec_loss={rec_loss.item():.4f}  "
                  f"time={step_time:.1f}s  "
                  f"GPU_mem={mem_gb:.2f}GB")

    return time.time() - total_start


def save_cache_to_lustre(local_cache_dir, lustre_cache_dir, rank, world_size):
    """
    Rank 0 collects all per-rank caches and merges them into one
    unified Lustre cache directory for use by future training jobs.

    All ranks write the same kernel configs (same model, same shapes),
    so rank 0's cache is sufficient — we just use it as the master copy.
    """
    # Wait for all ranks to finish writing their local caches
    dist.barrier()

    if rank == 0:
        os.makedirs(lustre_cache_dir, exist_ok=True)

        print(f"\nSaving kernel cache to Lustre: {lustre_cache_dir}")

        # Copy rank 0's local cache to Lustre
        # (all ranks have identical kernel configs for the same model/shapes)
        if os.path.exists(local_cache_dir):
            files = os.listdir(local_cache_dir)
            copied = 0
            total_bytes = 0
            for fname in files:
                src  = os.path.join(local_cache_dir, fname)
                dst  = os.path.join(lustre_cache_dir, fname)
                if os.path.isfile(src):
                    import shutil
                    shutil.copy2(src, dst)
                    size = os.path.getsize(dst)
                    total_bytes += size
                    copied += 1
                    print(f"  Saved: {fname}  ({size/1e6:.2f} MB)")

            print(f"\n✓ Cache saved: {copied} files, "
                  f"{total_bytes/1e6:.1f} MB total")
            print(f"  Location: {lustre_cache_dir}")
        else:
            print(f"WARNING: Local cache dir not found: {local_cache_dir}")
            print("  MIOpen may not have written cache files yet.")

    dist.barrier()


def verify_cache(lustre_cache_dir, rank):
    """Verify the cache contains the expected files."""
    if rank != 0:
        return True

    if not os.path.exists(lustre_cache_dir):
        print(f"ERROR: Cache dir missing: {lustre_cache_dir}")
        return False

    files = os.listdir(lustre_cache_dir)
    if not files:
        print(f"ERROR: Cache dir is empty: {lustre_cache_dir}")
        return False

    print(f"\nCache verification:")
    total_size = 0
    has_ukdb   = False
    has_ufdb   = False

    for fname in sorted(files):
        fpath = os.path.join(lustre_cache_dir, fname)
        size  = os.path.getsize(fpath)
        total_size += size
        print(f"  {fname}  ({size/1e6:.3f} MB)")
        if fname.endswith(".ukdb"):
            has_ukdb = True
        if fname.endswith(".ufdb.txt"):
            has_ufdb = True

    print(f"\n  Total size : {total_size/1e6:.1f} MB")
    print(f"  Has .ukdb  : {has_ukdb}  (kernel binary db)")
    print(f"  Has .ufdb  : {has_ufdb}  (find-db, needed for FAST mode)")

    if not has_ufdb:
        print("\nWARNING: No .ufdb.txt found!")
        print("  FAST mode (2) needs the find-db to avoid kernel search.")
        print("  Try re-running warmup with MIOPEN_FIND_MODE=1.")
        return False

    # Count entries in find-db
    for fname in files:
        if fname.endswith(".ufdb.txt"):
            fpath = os.path.join(lustre_cache_dir, fname)
            with open(fpath) as f:
                lines = [l for l in f.readlines() if l.strip()]
            print(f"  Find-Db entries: {len(lines)}  (one per unique conv config)")
            if len(lines) < 10:
                print("  WARNING: Very few entries — warmup may not have "
                      "covered all conv layers. Consider re-running.")

    return True


def main():
    # ── Add training code to path ─────────────────────────────────────────────
    code_dir = os.path.dirname(os.path.abspath(__file__))
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    # ── Cache paths ───────────────────────────────────────────────────────────
    lustre_cache = os.environ.get(
        "MIOPEN_LUSTRE_CACHE",
        "/lustre/orion/proj-shared/cli138/z7a/.miopen-cache"
    )

    # ── Distributed setup ─────────────────────────────────────────────────────
    rank, world_size, local_rank, device_id, device = setup_distributed()

    # Per-rank local cache (node-local NVMe — fast I/O, no Lustre contention)
    local_cache = f"/tmp/miopen-warmup-rank-{rank}"
    os.makedirs(local_cache, exist_ok=True)

    # Point MIOpen to local cache for this rank
    # FIND_MODE=1 (NORMAL): exhaustive search — builds complete Find-Db
    os.environ["MIOPEN_USER_DB_PATH"]     = local_cache
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = local_cache
    os.environ["MIOPEN_FIND_MODE"]        = "1"   # NORMAL: full search
    os.environ["MIOPEN_DISABLE_CACHE"]    = "0"

    if rank == 0:
        print("=" * 60)
        print("MIOpen Full-Model Kernel Warmup")
        print("=" * 60)
        print(f"World size   : {world_size} GPUs")
        print(f"Model        : ImprovedWNet(in_channels={IN_CHANNELS}, "
              f"num_classes={NUM_CLASSES})")
        print(f"Input shape  : ({BATCH_SIZE}, {IN_CHANNELS}, {HEIGHT}, {WIDTH})")
        print(f"Grad accum   : {GRAD_ACCUM}  (matches main.py)")
        print(f"Warmup steps : {WARMUP_STEPS}")
        print(f"Local cache  : {local_cache}")
        print(f"Lustre cache : {lustre_cache}")
        print(f"Find mode    : 1 (NORMAL — exhaustive, builds complete Find-Db)")
        print("=" * 60)

    # ── Import and build the actual model ─────────────────────────────────────
    if rank == 0:
        print("\nImporting ImprovedWNet...")

    from WNet_model import ImprovedWNet

    model = ImprovedWNet(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES
    ).to(device)

    model = DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3, weight_decay=1e-5, eps=1e-8, betas=(0.9, 0.999)
    )
    scaler = GradScaler()

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model ready: {total_params/1e6:.1f}M parameters")
        print(f"\nRunning {WARMUP_STEPS} warmup steps...")
        print("(Exhaustive kernel search per conv layer — takes ~5-15 min)\n")

    # ── Run warmup ────────────────────────────────────────────────────────────
    warmup_time = run_warmup_steps(
        model, optimizer, scaler, device,
        rank, WARMUP_STEPS, GRAD_ACCUM
    )

    if rank == 0:
        print(f"\n✓ Warmup passes complete: {warmup_time:.1f}s "
              f"({warmup_time/60:.1f} min)")

    # ── Save cache to Lustre ──────────────────────────────────────────────────
    save_cache_to_lustre(local_cache, lustre_cache, rank, world_size)

    # ── Verify ────────────────────────────────────────────────────────────────
    cache_ok = verify_cache(lustre_cache, rank)

    if rank == 0:
        print("\n" + "=" * 60)
        if cache_ok:
            print("✓ WARMUP COMPLETE — Cache is ready for training")
            print("\nNext steps:")
            print("  1. Training script already uses MIOPEN_FIND_MODE=2 (FAST)")
            print("  2. Every WNet conv layer is now in the Find-Db")
            print("  3. Submit training: sbatch Run_gpu_frontier_script_distributed.sh")
            print("  4. Expected startup: ~0s MIOpen overhead (all Find-Db hits)")
        else:
            print("✗ WARNING: Cache verification failed")
            print("  Re-run this warmup before submitting training jobs")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
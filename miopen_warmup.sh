#!/bin/bash -l
#SBATCH --account CLI138
#SBATCH -J miopen_warmup
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH -q normal
#SBATCH -o logs/warmup-%j.txt
#SBATCH -e logs/warmup-%j.txt
#SBATCH -C nvme
#SBATCH --export=NONE
# ─────────────────────────────────────────────────────────────────────────────
# MIOpen kernel warmup — runs a full WNet forward+backward pass at EXACT
# training shapes so every conv layer (all encoder/decoder levels) gets
# cached in the Find-Db.  Only needs to run once.
#
# After this completes, training uses MIOPEN_FIND_MODE=2 (FAST) and
# every layer is a Find-Db hit → zero search time.
#
# Key facts from main.py:
#   batch_size   = 2
#   in_channels  = 14
#   target_size  = (192, 384)
#   model        = ImprovedWNet(in_channels=14, num_classes=5)
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs

module load PrgEnv-gnu/8.7.0
module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

# ── MIOpen: NORMAL mode for warmup (exhaustive search, builds full Find-Db) ──
MIOPEN_LUSTRE_CACHE="/lustre/orion/proj-shared/cli138/z7a/.miopen-cache"
mkdir -p ${MIOPEN_LUSTRE_CACHE}

export MIOPEN_USER_DB_PATH="${MIOPEN_LUSTRE_CACHE}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_LUSTRE_CACHE}"
export MIOPEN_FIND_MODE=1          # NORMAL: exhaustive search, builds complete cache
export MIOPEN_DISABLE_CACHE=0
export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_LOG_LEVEL=0

export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=7
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

echo "=========================================="
echo "MIOpen Full-Model Kernel Warmup"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "Cache dir : ${MIOPEN_LUSTRE_CACHE}"
echo "Find mode : 1 (NORMAL — exhaustive, builds complete Find-Db)"
echo "Model     : ImprovedWNet(in_channels=14, num_classes=5)"
echo "Input     : (2, 14, 192, 384) — exact training shape"
echo "=========================================="
echo "Starting warmup at: $(date)"

mypython=$(which python3)

$mypython - << 'PYEOF'
import sys
import os
import time
import torch

# Add the training code directory to path so we can import the actual model
sys.path.insert(0, "/lustre/orion/cli138/proj-shared/z7a/Storms_data/"
                   "Storms_precipitation_notebooks/WNet_parallel_global_code")

from WNet_model import ImprovedWNet

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"Device count    : {torch.cuda.device_count()}")

device = torch.device("cuda:0")
torch.cuda.set_device(0)

# ── Exact training configuration from main.py ─────────────────────────────
BATCH_SIZE   = 2
IN_CHANNELS  = 14
NUM_CLASSES  = 5
HEIGHT       = 192
WIDTH        = 384
EPOCHS       = 40    # run enough passes to hit every code path
GRAD_ACCUM   = 3     # matches gradient_accumulation_steps in main.py

print(f"\nModel    : ImprovedWNet(in_channels={IN_CHANNELS}, num_classes={NUM_CLASSES})")
print(f"Input    : ({BATCH_SIZE}, {IN_CHANNELS}, {HEIGHT}, {WIDTH})")
print(f"Device   : {device} ({torch.cuda.get_device_name(0)})")

# ── Build model exactly as in main.py ─────────────────────────────────────
model = ImprovedWNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                               weight_decay=1e-5, eps=1e-8)
scaler = torch.cuda.amp.GradScaler()

# ── Run multiple passes so all encoder/decoder conv layers are warmed up ──
# WNet has 2 UNets: each has enc1..enc4 + bottleneck + dec1..dec4.
# All spatial scales (192x384 → 12x24) need to be exercised.
print(f"\nRunning {EPOCHS} warmup passes (exhaustive kernel search per layer)...")
print("This will take ~5-15 min. Progress shown per pass.\n")

total_start = time.time()

for ep in range(EPOCHS):
    ep_start = time.time()

    # Random input matching exact training shape
    x     = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH,
                        device=device, dtype=torch.float32)
    # Random targets
    seg_t = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, HEIGHT, WIDTH),
                          device=device)
    rec_t = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH,
                        device=device, dtype=torch.float32)

    # Simulate gradient accumulation loop (hits same layers multiple times)
    optimizer.zero_grad()
    for acc_step in range(GRAD_ACCUM):
        with torch.cuda.amp.autocast():
            seg_probs, rec = model(x)

            # Simple loss to force backward through all layers
            seg_loss = torch.nn.functional.cross_entropy(seg_probs, seg_t)
            rec_loss = torch.nn.functional.mse_loss(rec, rec_t)
            loss     = (seg_loss + rec_loss) / GRAD_ACCUM

        scaler.scale(loss).backward()

    scaler.step(optimizer)
    scaler.update()

    ep_time = time.time() - ep_start
    print(f"  Pass {ep+1:2d}/{EPOCHS}: loss={loss.item()*GRAD_ACCUM:.4f}  "
          f"time={ep_time:.1f}s  "
          f"GPU mem={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Force sync to ensure all kernels are benchmarked
    torch.cuda.synchronize()

total_time = time.time() - total_start
print(f"\n✓ Warmup complete: {total_time:.1f}s ({total_time/60:.1f} min)")

# ── Verify cache was written ───────────────────────────────────────────────
cache_dir = os.environ.get("MIOPEN_USER_DB_PATH", "")
if cache_dir and os.path.exists(cache_dir):
    files = os.listdir(cache_dir)
    total_size = sum(os.path.getsize(os.path.join(cache_dir, f))
                     for f in files) / 1e6
    print(f"✓ Cache written: {len(files)} files, {total_size:.1f} MB → {cache_dir}")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(cache_dir, f)) / 1e6
        print(f"    {f}  ({size:.2f} MB)")
else:
    print(f"WARNING: Cache dir not found or empty: {cache_dir}")

PYEOF

EXIT_CODE=$?
echo ""
echo "=========================================="
echo "Warmup completed at : $(date)"
echo "Cache location      : ${MIOPEN_LUSTRE_CACHE}"
echo "Exit code           : ${EXIT_CODE}"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✓ SUCCESS — training jobs will use cached kernels"
    echo "  Set MIOPEN_FIND_MODE=2 (FAST) in training script"
    ls -lh ${MIOPEN_LUSTRE_CACHE}/
else
    echo "✗ FAILED — check logs above"
fi
echo "=========================================="
exit $EXIT_CODE
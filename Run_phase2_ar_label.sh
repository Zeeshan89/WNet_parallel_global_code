#!/bin/bash -l
# Run_phase2_ar_label.sh — retrain WNet on the literature-based AR label (channel 6 of
# tensor_cache_v6_arlit); everything else identical to the shipped configuration.
#
#   sbatch --dependency=afterok:<baseline job> Run_phase2_ar_label.sh
#   sbatch Run_phase2_ar_label.sh 200 tensor_cache_v6_arlit G_arlit_e200
#
# $1 = epochs (default 200), $2 = cache (default tensor_cache_v6_arlit), $3 = tag.
#
# Why this run exists
# -------------------
# The shipped AR label is IVT > 400, one global scalar with no geometry and no direction
# test (dataset.py:629-633). The model reproduces it (AR IoU 0.97 against its own label),
# which is why the shipped AR product has saturated tropical bands (AR frequency 10.7 % in
# |lat| < 15 against 0.03-1.6 % for every established method), no ARs over land or poleward
# of 60 deg, and a doubled seasonal cycle. The new label (ar_label_lib.py, candidate L6 chosen
# by the user on 2026-09-21 from evaluations/ar_label_sweep/) is IVT above the per-cell
# 5-month 85th percentile with an absolute floor of 175 kg/m/s (100 poleward of 65), length
# >= 2000 km, length/width >= 2, mean poleward IVT > 50, coherent direction, and objects that
# fail are re-thresholded at the 87.5th to 95th percentile (Guan and Waliser 2015; Rutz 2014
# for the floor). Scored against the five established ERA5 AR methods before training, the
# label has tropical frequency 1.5 %, polar 5.0 %, W North America 4.4 % (shipped 0.6 %),
# pattern r 0.95 (shipped 0.41) and no area above 120 AR days per year (shipped 9.9 %).
#
# Hyperparameters are byte-identical to Run_phase2_tc_pwsweep.sh / F_ibtracs_pwH50_e200:
#   epochs=200 batch=8 ncut=0.1 ste alpha=0.5,0.3,0.4,0.45 beta=0.5,0.7,0.6,0.55
#   pos_weight=8,50,25,15 workers=4 accum=1 lr=0.0007 warmup=4 nodes=8 tasks=64
# Only the cache, the trivial-baseline floor of the AR class and the selected class differ.
# The AR product ships from best_ar.pth; TC / ETC / MCS keep their shipped checkpoints.
# The GPU run stays at 8 nodes on purpose: the world size sets the effective batch and the
# learning-rate schedule, so a wider run would not be the same experiment.

#SBATCH --account=CLI138
#SBATCH -J wnet_ar_lit
#SBATCH -p batch
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=01:45:00
#SBATCH -o logs/arlit-train-%j.txt
#SBATCH -e logs/arlit-train-err-%j.txt
#SBATCH --export=NONE

mkdir -p logs
unset SLURM_EXPORT_ENV

EPOCHS="${1:-200}"
CACHE="${2:-tensor_cache_v6_arlit}"
TAG="${3:-G_arlit_e${EPOCHS}}"
case "$EPOCHS" in ''|*[!0-9]*) echo "ERROR: epochs must be an integer, got '$EPOCHS'"; exit 1;; esac

module load PrgEnv-gnu/8.7.0
module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
mypython=$(which python3)
export PYTHONNOUSERSITE=1

export MIOPEN_FIND_MODE=2
export MIOPEN_DISABLE_CACHE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=7
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29561

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export SPLIT_CACHE_DIR="$NB/$CACHE"
CKPT="$NB/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2"
BASELINE="phase1_verify_${CACHE}_baseline.json"
PW="8,50,25,15"

n_months=$(ls "$SPLIT_CACHE_DIR"/split_*.npy 2>/dev/null | grep -vc "meta.npy")   # split_YYYYMM.meta.npy also matches
if [ "$n_months" -ne 540 ]; then echo "ERROR: $SPLIT_CACHE_DIR has $n_months/540 months"; exit 1; fi
if [ ! -s "$BASELINE" ]; then echo "ERROR: $BASELINE missing (run Run_ar_baseline.sh)"; exit 1; fi

echo "================================================================================"
echo " Phase 2 retrain on the literature-based AR label"
echo "   cache         = $SPLIT_CACHE_DIR ($n_months months)"
echo "   baseline      = $BASELINE: $(cat $BASELINE | tr -d '\n ' | cut -c1-120)"
echo "   epochs        = $EPOCHS   pos_weight = $PW [AR, Hurricane, ETC, MCS]"
echo "   select-class  = ar on iou (best_ar.pth); best.pth stays the 4-class choice"
echo "   ckpt          = $CKPT/$TAG"
echo "   nodes=$SLURM_NNODES tasks=$SLURM_NTASKS port=$MASTER_PORT  started $(date)"
echo "================================================================================"

srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
    $mypython -W ignore -u ./main_phase2.py \
        --epochs "$EPOCHS" \
        --batch-size 8 \
        --ncut-weight 0.1 \
        --discretize ste \
        --tversky-alpha 0.5,0.3,0.4,0.45 \
        --tversky-beta 0.5,0.7,0.6,0.55 \
        --max-pos-weight "$PW" \
        --num-workers 4 \
        --grad-accum 1 \
        --lr 0.0007 \
        --warmup-epochs 4 \
        --ckpt-dir "$CKPT" \
        --tag "$TAG" \
        --baseline-json "$BASELINE" \
        --select-class ar \
        --select-metric iou

RC=$?
echo "Finished $(date), exit $RC"

HIST="$CKPT/$TAG/history.json" $mypython - <<'PY'
import json, os
p = os.environ["HIST"]
try:
    h = json.load(open(p))
except Exception as e:
    print(f"\n  !! could not read {p}: {e}"); raise SystemExit(0)
cfg = h.get("config", h)
print(f"\n  max_pos_weight recorded = {cfg.get('max_pos_weight')!r}   baseline = {cfg.get('trivial_baseline')}")
hist = h.get("epochs", h.get("history", []))
if hist:
    b = max(hist, key=lambda r: r.get("val_ar_iou", -1))
    print(f"  best AR epoch {b['epoch']}: IoU {b['val_ar_iou']:.4f} R {b.get('val_ar_recall', float('nan')):.3f} "
          f"P {b.get('val_ar_precision', float('nan')):.3f}")
    last = hist[-1]
    print("  other heads at the last epoch (should match the shipped run, only the AR label changed): "
          + ", ".join(f"{c} {last.get(f'val_{c}_iou', float('nan')):.3f}" for c in ("hurricane", "etc", "mcs")))
    print("  reference F_ibtracs_pwH50_e200 best: AR 0.9712 (vs the OLD label), ETC 0.7625, MCS 0.8315, TC 0.5024")
PY
exit $RC

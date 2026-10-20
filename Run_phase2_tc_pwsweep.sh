#!/bin/bash -l
# Run_phase2_tc_pwsweep.sh — re-train the IBTrACS-target Phase 2 model with the TC
# head's pos_weight cap RESTORED, sweeping the cap value.
#
#   sbatch Run_phase2_tc_pwsweep.sh 50           # cap 50, 30 epochs (default)
#   sbatch --time=01:45:00 Run_phase2_tc_pwsweep.sh 50 200   # full-length run
#
# $1 = hurricane pos_weight cap, $2 = epochs (default 30). Bump --time when you
# bump the epochs: ~19 s/epoch, so 200 epochs is ~65 min of epoch time plus ~10 min
# of startup. The ckpt dir carries the epoch count, so runs never collide.
#
# The cap is taken as an ARGUMENT (arguments reach the script even under --export=NONE,
# unlike environment variables -- which is exactly how the original regression
# got in). Each arm writes its own ckpt dir, so all three can run concurrently.
#
# Why this run exists
# -------------------
# The shipped TC model (Phase2/E_ibtracs200/best.pth) has two defects, both
# measured, neither in the data:
#
#  1. pos_weight regression. E_ibtracs100 and E_ibtracs200 recorded
#     `max_pos_weight: 8.0` as a SCALAR; C_full80/C_full200/D_tierb200 all
#     recorded the config-C per-class list [8, 50, 25, 15]. Hurricane is class
#     index 1, so its BCE false-negative cap was 8 instead of 50. loss.py does
#     `pos_weight = clamp(n_neg/n_pos, 1, max_pos_weight[i])`, and the IBTrACS TC
#     label has a positive rate of ~0.00101, so n_neg/n_pos ~ 993 and the cap is
#     ALWAYS the binding constraint -- a straight 6.25x weakening of the FN
#     penalty on the sparsest head. TC recall duly falls 0.834 -> 0.648 while
#     precision climbs 0.160 -> 0.702.
#
#  2. Checkpoint selected on the wrong metric. best.pth maximises
#     val_skill_vs_baseline averaged over ALL FOUR classes. TC IoU peaks at
#     epoch 12 (0.5298, recall 0.697) and is still 0.5084 / 0.648 at the shipped
#     epoch 130, because AR/ETC/MCS gains outweigh the TC loss. `--select-class
#     hurricane` additionally writes best_hurricane.pth on TC IoU alone.
#     best.pth keeps its existing meaning, so the multi-class product is
#     unaffected.
#
# Why sweep rather than just set 50: pos_weight and Tversky beta both push
# recall and they interact (the first Phase 2 smoke run came out at recall ~1.0
# with useless precision). 50 is the config-C value, but the IBTrACS label is
# ~8x sparser than the CyTRACK label those numbers were tuned against, so the
# right cap for this label is an empirical question. Three arms cost the same
# wall clock as one.
#
# Independently measured and NOT the cause: the six predictors carry a full TC
# signature at 100% of the positions of the storms the model misses
# (phase4_tc_miss_learnability.py). This is a recall problem, not a missing input.
#
# Everything else is byte-identical to the E_ibtracs200 configuration, so any
# difference is attributable to the cap and the selection metric:
#   epochs=200 batch=8 months=0 ncut=0.1 ste alpha=0.5,0.3,0.4,0.45
#   beta=0.5,0.7,0.6,0.55 workers=4 accum=1 lr=0.0007 warmup=4 nodes=8 tasks=64
#
# --select-class hurricane matters most on a LONG run: best.pth converges the other
# three heads (a 30-epoch run left ETC IoU at 0.599 vs 0.792), while
# best_hurricane.pth captures the TC peak, which arrives early and then decays.
# One run therefore yields both a converged multi-class model and the best TC head.

#SBATCH --account=CLI138
#SBATCH -J wnet_tc_pw
#SBATCH -p batch
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:35:00
#SBATCH -o logs/tcpw-%j.txt
#SBATCH -e logs/tcpw-err-%j.txt
#SBATCH --export=NONE

mkdir -p logs
unset SLURM_EXPORT_ENV

HCAP="${1:?usage: sbatch Run_phase2_tc_pwsweep.sh <cap> [epochs] [metric] [minprec]}"
EPOCHS="${2:-30}"
SELMETRIC="${3:-iou}"
SELMINP="${4:-0}"
for v in "$HCAP" "$EPOCHS"; do
  case "$v" in ''|*[!0-9]*) echo "ERROR: expected integers, got '$v'"; exit 1;; esac
done
case "$SELMETRIC" in iou|recall) ;; *) echo "ERROR: metric must be iou|recall"; exit 1;; esac

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
# Distinct per arm so three concurrent jobs cannot collide on a shared node.
export MASTER_PORT=$((29500 + HCAP))

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
# The IBTrACS-target label build (channel 7 = ROCI discs). Same cache as E_ibtracs200.
export SPLIT_CACHE_DIR="$NB/tensor_cache_v5_ibtracs"
CKPT="$NB/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2"
TAG="F_ibtracs_pwH${HCAP}_e${EPOCHS}"
PW="8,${HCAP},25,15"

echo "================================================================================"
echo " Phase 2 TC retrain: pos_weight cap restored + TC-selected checkpoint"
echo "   hurricane cap = $HCAP   (shipped model ran with a scalar 8)"
echo "   epochs        = $EPOCHS"
echo "   pos_weight    = $PW   [AR, Hurricane, ETC, MCS]"
echo "   select-class  = hurricane on $SELMETRIC (min precision $SELMINP)"
echo "   cache         = $SPLIT_CACHE_DIR"
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
        --select-class hurricane \
        --select-metric "$SELMETRIC" \
        --select-min-precision "$SELMINP"

RC=$?
echo "Finished $(date), exit $RC"

# The whole point of the run: confirm the per-class list actually landed. A scalar
# here means the hurricane head was mis-weighted again and the arm is void.
HIST="$CKPT/$TAG/history.json" SELMETRIC="$SELMETRIC" SELMINP="$SELMINP" \
  $mypython - <<'PY'
import json, os
p = os.environ["HIST"]
try:
    h = json.load(open(p))
except Exception as e:
    print(f"\n  !! could not read {p}: {e}")
    raise SystemExit(0)
cfg = h.get("config", h)
mpw = cfg.get("max_pos_weight")
print(f"\n  max_pos_weight recorded = {mpw!r}")
print("  VERDICT:", "OK - per-class list landed" if isinstance(mpw, list)
      else "VOID - scalar again, hurricane head mis-weighted")
hist = h.get("history", [])
if hist:
    _mt = os.environ.get("SELMETRIC", "iou")
    _fl = float(os.environ.get("SELMINP", "0") or 0)
    if _mt == "recall":
        _ok = [r for r in hist
               if r.get("val_hurricane_precision", 0) >= _fl]
        b = max(_ok or hist, key=lambda r: r.get("val_hurricane_recall", -1))
    else:
        b = max(hist, key=lambda r: r.get("val_hurricane_iou", -1))
    print(f"  best TC epoch {b['epoch']}: IoU {b['val_hurricane_iou']:.4f} "
          f"R {b['val_hurricane_recall']:.3f} P {b['val_hurricane_precision']:.3f}")
    print( "  reference E_ibtracs200 epoch 130 (shipped): IoU 0.5084 R 0.648 P 0.702")
    print( "  reference E_ibtracs200 epoch  12 (TC opt) : IoU 0.5298 R 0.697 P 0.688")
PY
exit $RC

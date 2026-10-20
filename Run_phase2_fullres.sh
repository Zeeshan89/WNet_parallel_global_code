#!/bin/bash -l
# Run_phase2_fullres.sh — WNet trained at native ERA5 0.25 deg (tensor_cache_v11_fullres, 721 x 1440,
# cropped to 720 rows), same labels as the final shipped models (AR L6, ETC cc, MCS legacy; TC
# lifecycle + TD = v9), same loss and selection as Run_phase2_final.sh. Self-chaining chunks.
#
#   sbatch Run_phase2_fullres.sh                                  # defaults below
#   sbatch --export=ALL,INIT=,TAG=J_fullres_scratch Run_phase2_fullres.sh      # from scratch
#
# What differs from the 192 x 384 run (Run_phase2_final.sh), and why:
#   --crop-rows 720          the U-Net pools 3x, so H must divide by 8; drops the -90 row
#   --ncut-radius/sigma1 15  the NCut kernel is in pixels: 4 px at 0.94 deg = 15 px at 0.25 deg
#   --batch-size 2           per GPU (probe_fullres_memory.py: 728x1440 fits bs 2-4 per 64 GB GCD);
#                            64 GPUs x 2 = 128 samples/step, each ~14x the pixels of a 192x384 one
#   --stats predictor_stats_fullres.json   native fields have a wider spread than the bilinearly
#                            reduced ones; computed once from this cache (prelude below)
#   INIT                     default: start from H_final_e300/final.pth (the shipped 0.94 deg model)
#                            with a lower LR; empty INIT trains from scratch
#   --train-eval-stride 16   the per-epoch training-set scoring is 14x dearer per sample
# Checkpoints as before: best.pth, best_ar/_hurricane/_etc/_mcs.pth, last.pth, final.pth.
#SBATCH --account=CLI138
#SBATCH -J wnet_fullres
#SBATCH -p batch
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=01:00:00
#SBATCH -o logs/fullres-train-%j.txt
#SBATCH -e logs/fullres-train-err-%j.txt

mkdir -p logs
unset SLURM_EXPORT_ENV

EPOCHS_TOTAL="${EPOCHS_TOTAL:-60}"
EPOCHS_PER_JOB="${EPOCHS_PER_JOB:-100}"
CACHE="${CACHE:-tensor_cache_v11_fullres}"
TAG="${TAG:-J_fullres_e${EPOCHS_TOTAL}}"
INIT="${INIT-ARs_Hurricanes_ETCs_MCS/Phase2/H_final_e300/final.pth}"
LR="${LR:-$([ -n "$INIT" ] && echo 0.0003 || echo 0.0007)}"
BS="${BS:-2}"
CHUNK="${CHUNK:-1}"
MAX_CHUNKS="${MAX_CHUNKS:-30}"
STATS="predictor_stats_fullres.json"

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
export MIOPEN_FIND_MODE=2 MIOPEN_DISABLE_CACHE=0 HSA_FORCE_FINE_GRAIN_PCIE=1 OMP_NUM_THREADS=7
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29573

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export SPLIT_CACHE_DIR="$NB/$CACHE"
CKPT="$NB/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2"
BASELINE="phase1_verify_${CACHE}_baseline.json"
PW="8,50,25,15"

# every month present and non-empty (a 0-byte placeholder would crash SplitCacheDataset)
n_months=$(find "$SPLIT_CACHE_DIR" -maxdepth 1 -name 'split_??????.npy' -size +0 | wc -l)
[ "$n_months" -eq 540 ] || { echo "ERROR: $SPLIT_CACHE_DIR has $n_months/540 non-empty months"; exit 1; }
[ -s "$BASELINE" ] || { echo "ERROR: $BASELINE missing"; exit 1; }
[ -z "$INIT" ] || [ -s "$INIT" ] || { echo "ERROR: INIT $INIT missing"; exit 1; }

# predictor standardisation for the native fields, once (same method as predictor_stats.json)
if [ ! -s "$STATS" ]; then
  echo "computing $STATS from $CACHE  $(date)"
  srun -N1 -n1 -c 56 --export=ALL $mypython -u compute_predictor_stats.py --cache-dir "$SPLIT_CACHE_DIR" \
       --moments --out "$STATS.tmp" && mv "$STATS.tmp" "$STATS" || { echo "ERROR: stats failed"; exit 1; }
fi

# one chunk at a time: the same chunk may be submitted to two queues (debug + normal), whichever starts
# first trains. Exit if another chunk is running; cancel waiting siblings that depend on nothing.
others=$(squeue -u "$USER" -h -n wnet_fullres -t RUNNING -o %i | grep -vx "$SLURM_JOB_ID")
if [ -n "$others" ]; then echo "another wnet_fullres chunk is running ($others); this one exits"; exit 0; fi
for j in $(squeue -u "$USER" -h -n wnet_fullres -t PENDING -o "%i|%E" | awk -F'|' '$2=="" || $2=="(null)" {print $1}'); do
  scancel "$j" && echo "cancelled waiting sibling chunk $j"
done

RESUME=""
done_epochs=0
if [ -s "$CKPT/$TAG/last.pth" ]; then
  RESUME="--resume $CKPT/$TAG/last.pth"
  done_epochs=$(HIST="$CKPT/$TAG/history.json" $mypython -c '
import json, os
try: print(len(json.load(open(os.environ["HIST"])).get("epochs", [])))
except Exception: print(0)')
fi
if [ "$done_epochs" -ge "$EPOCHS_TOTAL" ]; then echo "$TAG already has $done_epochs epochs; nothing to do"; exit 0; fi
# no-progress guard: stop the chain if the previous chunk added no epoch (2026-09-26: 27 chunks in a row
# died in epoch 1 on /dev/shm and each queued the next)
PREV_EPOCHS="${PREV_EPOCHS:--1}"
if [ "$CHUNK" -gt 1 ] && [ "$done_epochs" -le "$PREV_EPOCHS" ]; then
  echo "ERROR: previous chunk made no progress ($done_epochs epochs, was $PREV_EPOCHS); not training, chain stopped"; exit 1
fi
# --init-weights only on a fresh start; a resumed chunk restores everything from last.pth
INIT_ARG=""
[ -z "$RESUME" ] && [ -n "$INIT" ] && INIT_ARG="--init-weights $INIT"

# successor queued now with afterany, so a walltime kill does not break the chain
if [ "$CHUNK" -lt "$MAX_CHUNKS" ]; then
  for i in 1 2 3 4 5; do
    NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID \
           --export=ALL,EPOCHS_TOTAL=$EPOCHS_TOTAL,EPOCHS_PER_JOB=$EPOCHS_PER_JOB,CACHE=$CACHE,TAG=$TAG,INIT=$INIT,LR=$LR,BS=$BS,CHUNK=$((CHUNK + 1)),MAX_CHUNKS=$MAX_CHUNKS,PREV_EPOCHS=$done_epochs \
           "$NB/WNet_parallel_global_code/Run_phase2_fullres.sh" 2>&1) && case "$NEXT" in *[!0-9]*) ;; *) echo "chunk $((CHUNK + 1)) queued as $NEXT (afterany:$SLURM_JOB_ID)"; break;; esac
    echo "  queue attempt $i: $NEXT"; sleep 20
  done
fi

echo "================================================================================"
echo " FULL-RES (0.25 deg) WNet $TAG: epochs $done_epochs -> of $EPOCHS_TOTAL   $(date)"
echo "   cache = $SPLIT_CACHE_DIR   stats = $STATS   init = ${INIT_ARG:-none}   resume = ${RESUME:-none}"
echo "   lr = $LR  batch/GPU = $BS  nodes = $SLURM_NNODES  tasks = $SLURM_NTASKS  pos_weight = $PW"
echo "================================================================================"

srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
    $mypython -W ignore -u ./main_phase2.py \
        --epochs "$EPOCHS_TOTAL" --epochs-this-run "$EPOCHS_PER_JOB" $RESUME $INIT_ARG \
        --batch-size "$BS" --ncut-weight 0.1 --ncut-radius 15 --ncut-sigma1 15 --crop-rows 720 \
        --discretize ste --stats "$STATS" \
        --tversky-alpha 0.5,0.3,0.4,0.45 --tversky-beta 0.5,0.7,0.6,0.55 \
        --max-pos-weight "$PW" --num-workers 4 --grad-accum 1 --lr "$LR" --warmup-epochs 2 \
        --val-stride 4 --train-eval-stride 16 \
        --ckpt-dir "$CKPT" --tag "$TAG" --baseline-json "$BASELINE" \
        --select-class ar hurricane etc mcs \
        --select-metric iou recall iou iou \
        --select-min-precision "${TC_PRECISION_FLOOR:-0.60}"
RC=$?
echo "Finished $(date), exit $RC"
exit $RC

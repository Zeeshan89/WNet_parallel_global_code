#!/bin/bash -l
# Run_phase2_tclife.sh — short TC-focused retrain on the FULL-LIFECYCLE TC label
# (tensor_cache_v8_tclife = v7 with only channel 7 replaced; see Run_build_tclife_label.sh).
# Copy of Run_phase2_final.sh (identical hyperparameters, chunking and self-chaining) with:
#   * CACHE default tensor_cache_v8_tclife, EPOCHS_TOTAL default 40 (the TC head peaks at
#     epoch 8-13 in every run so far; 300 epochs only degrade it)
#   * INIT=<ckpt> passes --init-weights (ignored on resume, which restores everything)
#   * only best_hurricane.pth (recall at precision >= 0.60) is used downstream; AR/ETC/MCS
#     products are NOT re-exported from this run.
#   sbatch --export=ALL,TAG=I_tclife_ft40,INIT=$PWD/ARs_Hurricanes_ETCs_MCS/Phase2/H_final_e300_single/final.pth Run_phase2_tclife.sh
#   sbatch --export=ALL,TAG=I_tclife_scratch40 Run_phase2_tclife.sh
#SBATCH --account=CLI138
#SBATCH -J wnet_tclife
#SBATCH -p batch
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH -o logs/tclife-train-%j.txt
#SBATCH -e logs/tclife-train-err-%j.txt

mkdir -p logs
unset SLURM_EXPORT_ENV

EPOCHS_TOTAL="${EPOCHS_TOTAL:-40}"
EPOCHS_PER_JOB="${EPOCHS_PER_JOB:-40}"
CACHE="${CACHE:-tensor_cache_v8_tclife}"
INIT="${INIT:-}"
TAG="${TAG:-I_tclife_e${EPOCHS_TOTAL}}"
CHUNK="${CHUNK:-1}"                 # position in the chain, for the safety cap below
MAX_CHUNKS="${MAX_CHUNKS:-3}"

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
export MASTER_PORT=29571

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export SPLIT_CACHE_DIR="$NB/$CACHE"
CKPT="$NB/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2"
BASELINE="phase1_verify_${CACHE}_baseline.json"
PW="8,50,25,15"

n_months=$(ls "$SPLIT_CACHE_DIR"/split_*.npy 2>/dev/null | grep -vc "meta.npy")
[ "$n_months" -eq 540 ] || { echo "ERROR: $SPLIT_CACHE_DIR has $n_months/540 months"; exit 1; }
[ -s "$BASELINE" ] || { echo "ERROR: $BASELINE missing (Run_ar_baseline.sh CLASSES='ar etc' CACHE=$CACHE)"; exit 1; }

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

# The successor is queued NOW with afterany, not at the end of this job: chunk 1 was killed by
# the 45-minute limit while still inside its epoch loop (start-up on the fresh v7 cache took 28
# of the 45 minutes), so an end-of-job resubmission never ran. With afterany the chain survives
# a walltime kill, and the guard above makes the last queued chunk exit immediately once
# EPOCHS_TOTAL epochs are recorded. MAX_CHUNKS stops a broken run from looping forever.
if [ "$CHUNK" -lt "$MAX_CHUNKS" ]; then
  for i in 1 2 3 4 5; do
    NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID \
           --export=ALL,EPOCHS_TOTAL=$EPOCHS_TOTAL,EPOCHS_PER_JOB=$EPOCHS_PER_JOB,CACHE=$CACHE,TAG=$TAG,INIT=$INIT,CHUNK=$((CHUNK + 1)),MAX_CHUNKS=$MAX_CHUNKS \
           "$NB/WNet_parallel_global_code/Run_phase2_tclife.sh" 2>&1) && case "$NEXT" in *[!0-9]*) ;; *) echo "chunk $((CHUNK + 1)) queued as $NEXT (afterany:$SLURM_JOB_ID)"; break;; esac
    echo "  queue attempt $i: $NEXT"; sleep 20
  done
else
  echo "CHUNK $CHUNK reached MAX_CHUNKS $MAX_CHUNKS; not queueing a successor"
fi

echo "================================================================================"
echo " TC-lifecycle retrain $TAG: epochs $done_epochs -> at most $((done_epochs + EPOCHS_PER_JOB)) of $EPOCHS_TOTAL this job"
echo "   cache    = $SPLIT_CACHE_DIR   baseline = $BASELINE: $(tr -d '\n ' < $BASELINE | cut -c1-110)"
echo "   resume   = ${RESUME:-none}   pos_weight = $PW   nodes=$SLURM_NNODES tasks=$SLURM_NTASKS   $(date)"
echo "================================================================================"

srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
    $mypython -W ignore -u ./main_phase2.py \
        --epochs "$EPOCHS_TOTAL" --epochs-this-run "$EPOCHS_PER_JOB" $RESUME \
        --batch-size 8 --ncut-weight 0.1 --discretize ste \
        --tversky-alpha 0.5,0.3,0.4,0.45 --tversky-beta 0.5,0.7,0.6,0.55 \
        --max-pos-weight "$PW" --num-workers 4 --grad-accum 1 --lr 0.0007 --warmup-epochs 4 \
        --ckpt-dir "$CKPT" --tag "$TAG" --baseline-json "$BASELINE" \
        --select-class ar hurricane etc mcs \
        --select-metric iou recall iou iou \
        --select-min-precision "${TC_PRECISION_FLOOR:-0.60}" \
        ${INIT:+--init-weights $INIT}
RC=$?
echo "Finished $(date), exit $RC"

now_epochs=$(HIST="$CKPT/$TAG/history.json" $mypython -c '
import json, os
try: print(len(json.load(open(os.environ["HIST"])).get("epochs", [])))
except Exception: print(0)')
echo "epochs recorded: $now_epochs / $EPOCHS_TOTAL"
if [ "$now_epochs" -ge "$EPOCHS_TOTAL" ]; then
  HIST="$CKPT/$TAG/history.json" $mypython - <<'PY'
import json, os
h = json.load(open(os.environ["HIST"])); e = h.get("epochs", [])
for c in ("ar", "hurricane", "etc", "mcs"):
    b = max(e, key=lambda r: r.get(f"val_{c}_iou", -1))
    print(f"  {c:<10} best val IoU {b[f'val_{c}_iou']:.4f} at epoch {b['epoch']}")
print("  export AR from best_ar.pth, ETC from best.pth (Run_export_ar_label.sh / Run_export_pw50_e200.sh pattern)")
PY
fi
exit $RC

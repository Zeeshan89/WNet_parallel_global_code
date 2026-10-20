#!/bin/bash -l
# Run_phase2_final.sh — the final WNet retrain on the new AR label (L6) and the new ETC label
# (closed-low centres), 300 epochs, run as SELF-CHAINING 45-minute chunks.
#
#   sbatch Run_phase2_final.sh                                   # defaults below
#   sbatch --export=ALL,CACHE=tensor_cache_v7_arlit_etccc,TAG=H_final_e300 Run_phase2_final.sh
#
# Why chunks. A 300-epoch run costs about 10 min of startup plus 19.5 s per epoch on 8 nodes
# (110 min); the user wants short jobs, which start far sooner in the queue (a 30-minute request
# backfilled two hours ahead of a 1:57 one on 2026-09-22), so --time is 30 minutes here.
# Start-up dominates such a chunk: on the freshly written v7 cache chunk 1 spent 28 of its 45
# minutes before the first epoch (cold Lustre metadata for 540 files across 64 ranks) and
# managed 46 epochs at 20 s each. Raise --time again if a chunk adds too few epochs to be worth
# its start-up. main_phase2.py's --epochs-this-run stops cleanly after N epochs
# and --resume last.pth restores model, optimizer, scheduler, EMA and history, so the cosine
# schedule (built from the FULL --epochs) is identical to an unchunked run. Each job resumes
# if last.pth exists, runs at most EPOCHS_PER_JOB epochs, and resubmits itself until
# history.json holds EPOCHS_TOTAL epochs. Checkpoints: best.pth (4-class skill vs the floors),
# best_ar.pth (AR IoU), last.pth (every epoch), final.pth.
#
# Checkpoints: best.pth (4-class skill) plus one per storm type, each on the criterion that
# suits it. AR, ETC and MCS are selected on their own IoU; TC is selected on RECALL subject to
# a precision floor (TC_PRECISION_FLOOR, default 0.60), because IoU is measurably the wrong
# criterion there: on an earlier run max TC IoU chose an epoch with recall 0.670 over one with
# 0.754, and tracked those gave POD 0.808 against 0.850. The chained H_final_e300 run selected
# on AR alone, so its TC class had to come from best.pth and the tracker found 8 fewer storms
# per year than the v2 product; this run fixes that.
#
# Prerequisites (both checked): the cache has 540 months and phase1_verify_<CACHE>_baseline.json
# exists (Run_ar_baseline.sh with CLASSES="ar etc" on the new cache: the AR and ETC floors
# change with their labels; TC / MCS keep the shipped floors).
# Hyperparameters are byte-identical to the shipped runs (Run_phase2_tc_pwsweep.sh).

#SBATCH --account=CLI138
#SBATCH -J wnet_final
#SBATCH -p batch
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH -o logs/final-train-%j.txt
#SBATCH -e logs/final-train-err-%j.txt

mkdir -p logs
unset SLURM_EXPORT_ENV

EPOCHS_TOTAL="${EPOCHS_TOTAL:-300}"
EPOCHS_PER_JOB="${EPOCHS_PER_JOB:-100}"
CACHE="${CACHE:-tensor_cache_v7_arlit_etccc}"
TAG="${TAG:-H_final_e${EPOCHS_TOTAL}}"
CHUNK="${CHUNK:-1}"                 # position in the chain, for the safety cap below
MAX_CHUNKS="${MAX_CHUNKS:-12}"

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
           --export=ALL,EPOCHS_TOTAL=$EPOCHS_TOTAL,EPOCHS_PER_JOB=$EPOCHS_PER_JOB,CACHE=$CACHE,TAG=$TAG,CHUNK=$((CHUNK + 1)),MAX_CHUNKS=$MAX_CHUNKS \
           "$NB/WNet_parallel_global_code/Run_phase2_final.sh" 2>&1) && case "$NEXT" in *[!0-9]*) ;; *) echo "chunk $((CHUNK + 1)) queued as $NEXT (afterany:$SLURM_JOB_ID)"; break;; esac
    echo "  queue attempt $i: $NEXT"; sleep 20
  done
else
  echo "CHUNK $CHUNK reached MAX_CHUNKS $MAX_CHUNKS; not queueing a successor"
fi

echo "================================================================================"
echo " FINAL retrain $TAG: epochs $done_epochs -> at most $((done_epochs + EPOCHS_PER_JOB)) of $EPOCHS_TOTAL this job"
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
        --select-min-precision "${TC_PRECISION_FLOOR:-0.60}"
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

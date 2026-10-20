#!/bin/bash -l
# Run_derisk_andes_cpu.sh — Phase-0 zero-fill de-risk on ANDES CPU (no GPU / no Frontier queue).
# ONE job runs BOTH halves of the gate for JJA 2004:
#   Wave 1  MCS_ONLY=0  -> derisk_full     (baseline: all 14 channels live)
#   Wave 2  MCS_ONLY=1  -> derisk_mcsonly  (ch0-10 zeroed, only MCS ch11-13 live)
# Each wave shards its 3 monthly files across 3 tasks (test.py get_rank_files),
# ~10 CPU cores/task. test.py auto-selects CPU (torch.cuda.is_available()==False).
# Waves run sequentially so at most 3 processes are resident (bounded memory).
#
# Submit:  sbatch -M andes Run_derisk_andes_cpu.sh

#SBATCH -A cli138
#SBATCH -J wnet_derisk_cpu
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=10
#SBATCH -t 04:00:00
#SBATCH -o logs/derisk_cpu-%j.txt
#SBATCH -e logs/derisk_cpu-err-%j.txt

set -u
CODE=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code
BASE=$CODE/ARs_Hurricanes_ETCs_MCS
CKPT=$BASE/Trail9/final_checkpoint_2006_2024.pth
DATA=/lustre/orion/cli138/proj-shared/z7a/Storms_data/data/
PY=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/bin/python

cd "$CODE"
mkdir -p logs

# CPU threading: one BLAS/OMP pool per task = cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export LD_LIBRARY_PATH=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1

echo "=============================================================="
echo " Phase-0 de-risk on ANDES CPU"
echo "  Job ID     : ${SLURM_JOB_ID:-(local)}   Node: $SLURM_NODELIST"
echo "  Tasks      : $SLURM_NTASKS x $OMP_NUM_THREADS threads"
echo "  Checkpoint : $CKPT"
echo "  Window     : 2004  months 6-8   (FORCE_REPROCESS)"
echo "  Started    : $(date)"
echo "=============================================================="

run_wave () {
  local mode=$1 outdir=$2
  echo ""
  echo ">>> WAVE MCS_ONLY=$mode -> $outdir   ($(date))"
  mkdir -p "$outdir"
  srun -N 1 -n 3 -c "$OMP_NUM_THREADS" --export=ALL,MCS_ONLY=$mode \
      "$PY" -W ignore -u ./test.py \
        --checkpoint  "$CKPT" \
        --data-dir    "$DATA" \
        --output-dir  "$outdir" \
        --start-year 2004 --end-year 2004 \
        --start-month 6   --end-month 8 \
        --force-reprocess
  echo ">>> WAVE MCS_ONLY=$mode done rc=$? ($(date))"
  echo "    MCS files: $(find "$outdir" -name '*_MCS_detection.nc' | wc -l)"
}

run_wave 0 "$BASE/derisk_full"
run_wave 1 "$BASE/derisk_mcsonly"

echo ""
echo "=============================================================="
echo "  Finished   : $(date)"
for d in derisk_full derisk_mcsonly; do
  printf "  %-16s MCS=%s\n" "$d" "$(find "$BASE/$d" -name '*_MCS_detection.nc' | wc -l)"
done
echo "=============================================================="

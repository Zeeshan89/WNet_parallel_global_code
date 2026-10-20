#!/bin/bash -l
# Run_hourly_split_cache.sh — build the HOURLY predictor cache for MCS inference.
#
# Produces (N, 6, 192, 384) tensors holding only the 6 raw physical predictors.
# The 4 heuristic label channels are deliberately absent: inference never reads
# them (phase4_export_detections.py:160 slices `arr[sel, :6]`), so an hourly
# CyTRACK / IBTrACS / wind-shear rebuild is not needed. That also keeps the cache
# at ~650 GB instead of ~1.05 TB.
#
# BT and IWV are real hourly fields; IVT, MSLP, vorticity and wind850 are linearly
# interpolated from the 6-hourly cache, because the MCS head is >96% insensitive to
# all four (phase1_diagnostics/phase1_results.json leave-one-out) and hourly
# vorticity/wind850 are not available on disk at 0.25 deg hourly anyway.
#
# Verified for 2004-07 against tensor_cache_v3_split at the shared 6-hourly stamps:
#   ch0-3 exact (MAE 0) | BT corr 1.000000, MAE 0.0041 K | IWV corr 0.998721
#
# SPLIT INTO THREE 15-YEAR JOBS, not one 45-year job. A 45 min+ Frontier request
# can sit in the queue for hours while the same work at ~30 min backfills almost
# immediately, so three short jobs finish sooner in wall-clock than one long one.
# Each job is 180 months over 64 ranks = 2-3 months per rank. With the per-year BT
# handle reuse in build_hourly_split_cache.py a rank touches at most 2 of the 36 GB
# BT files, so ~600 s for its first month and ~450 s after: ~25 min.
#
# Resumable and safe to resubmit: a month whose .npy already exists is skipped,
# and saves go through a temp file + rename so a walltime kill cannot leave a
# truncated month that the skip test would accept as complete.
#
# Submit (three jobs, they are independent and can run concurrently):
#   sbatch --export=YEARS="$(seq 1980 1994)" Run_hourly_split_cache.sh
#   sbatch --export=YEARS="$(seq 1995 2009)" Run_hourly_split_cache.sh
#   sbatch --export=YEARS="$(seq 2010 2024)" Run_hourly_split_cache.sh
#
#   sbatch --export=YEARS="2004" Run_hourly_split_cache.sh          # single-year pilot

#SBATCH --account=CLI138
#SBATCH -J wnet_hourly_cache
#SBATCH -p batch
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:35:00
#SBATCH -o logs/hourlycache-%j.txt
#SBATCH -e logs/hourlycache-err-%j.txt
#SBATCH --export=NONE

mkdir -p logs
unset SLURM_EXPORT_ENV

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
export OMP_NUM_THREADS=7

YEARS="${YEARS:-2004}"
OUTDIR="${OUTDIR:-/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/tensor_cache_hourly}"
DYNAMICS="${DYNAMICS:-interp}"

echo "================================================================================"
echo " Hourly predictor-cache build (MCS inference)"
echo "   years    : ${YEARS}"
echo "   outdir   : ${OUTDIR}"
echo "   dynamics : ${DYNAMICS}"
echo "   ranks    : $SLURM_NTASKS"
echo "   started  : $(date)"
echo "================================================================================"

srun -n $SLURM_NTASKS -c 7 --export=ALL \
    $mypython -W ignore -u ./build_hourly_split_cache.py \
        --years $YEARS \
        --outdir "$OUTDIR" \
        --dynamics "$DYNAMICS"

RC=$?
echo "Finished at $(date), exit $RC"
exit $RC

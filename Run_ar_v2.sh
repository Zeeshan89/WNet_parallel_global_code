#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J ar_v2
#SBATCH -p batch
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH -o logs/arv2-%j.txt
#SBATCH -e logs/arv2-err-%j.txt
#SBATCH --export=NONE
#
# Build an AR-vs-MARS evaluation cache. Which product is scored is set by two
# environment variables, so the same pass builds both the v2 and the v1 caches
# and the figure rows are then comparable:
#
#   sbatch Run_ar_v2.sh                                   # v2 (default)
#   sbatch --export=ALL,AR_PRED_DIR=...,AR_OUTDIR=... Run_ar_v2.sh
#
# 30 min: the v2 pass took ~2 min at 32 ranks, and the extra accumulators plus
# the larger uncompressed v1 files still leave plenty of headroom. Short jobs
# also clear the Frontier queue much faster.

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
PRED_DIR=${AR_PRED_DIR:-$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2/ARs_WNet_final}
OUTDIR=${AR_OUTDIR:-evaluations/ar_mars_eval_v2}
echo "pred-dir : $PRED_DIR"
echo "outdir   : $OUTDIR"

srun -n $SLURM_NTASKS -c 7 --export=ALL \
  $(which python3) -W ignore -u ./phase4_ar_vs_mars_v2.py \
    --pred-dir "$PRED_DIR" --outdir "$OUTDIR"
echo "exit $?"; ls $OUTDIR/part_*.npz 2>/dev/null | wc -l

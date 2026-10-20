#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J ar_obj_v2
#SBATCH -p batch
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH -o logs/arobj-%j.txt
#SBATCH -e logs/arobj-err-%j.txt
#SBATCH --export=NONE
#
# ARTMIP-style object-overlap AR metrics. The v1 equivalent ran serially for
# about 68 minutes; sharded over 32 ranks this is a few minutes.
#
#   sbatch Run_ar_objects_v2.sh
#   sbatch --export=ALL,AR_PRED_DIR=...,AR_OUTDIR=... Run_ar_objects_v2.sh

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
  $(which python3) -W ignore -u ./phase4_ar_objects_v2.py \
    --pred-dir "$PRED_DIR" --outdir "$OUTDIR"
echo "exit $?"; ls $OUTDIR/objpart_*.json 2>/dev/null | wc -l

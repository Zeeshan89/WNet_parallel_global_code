#!/bin/bash -l
# Run_export_v2.sh — full-record inference export, 1980-2024, all four classes.
# Writes Global_ARs_Hurr_ETCs_MCS_finalized_v2/ mirroring the old layout.
# Resumable: existing months are skipped, so just resubmit if it hits the wall.
#SBATCH --account=CLI138
#SBATCH -J wnet_expv2
#SBATCH -p batch
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:25:00
#SBATCH -o logs/expv2-%j.txt
#SBATCH -e logs/expv2-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7
export MIOPEN_FIND_MODE=2 MIOPEN_DISABLE_CACHE=0
OUT=${OUT:-/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/Global_ARs_Hurr_ETCs_MCS_finalized_v2}
# Checkpoint as a variable so the job log records which weights produced the output.
# C_full200 (200 epochs, best.pth = epoch 157) supersedes C_full80 (epoch 73):
# mean val IoU 0.759 vs 0.745, better on all four classes.
CKPT=${CKPT:-ARs_Hurricanes_ETCs_MCS/Phase2/C_full200/best.pth}
YEARS=$(seq 1980 2024 | tr '\n' ' ')
echo "Exporting 1980-2024, all 4 classes, 6-HOURLY"
echo "  checkpoint : $CKPT"
echo "  cache      : (default split cache, 6-hourly)"
echo "  outdir     : $OUT"
srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_export_detections.py \
    --checkpoint "$CKPT" \
    --years $YEARS --classes AR Hurricane ETC MCS \
    --subdirs --outdir $OUT
RC=$?; echo "exit $RC"; du -sh $OUT 2>/dev/null
echo "files: $(ls $OUT/*/ 2>/dev/null | grep -c '\.nc')/2160"
exit $RC

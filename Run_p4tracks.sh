#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J wnet_p4trk
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=01:00:00
#SBATCH -o logs/p4trk-%j.txt
#SBATCH -e logs/p4trk-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7
BASE=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2_detections
YEARS="1983 1994 2003 2012 2021"
echo "### 1) export detections"
srun -n1 -c7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_export_detections.py \
    --checkpoint ARs_Hurricanes_ETCs_MCS/Phase2/C_full80/best.pth \
    --years $YEARS --classes Hurricane ETC --outdir $BASE/nc
echo "### 2) run trackers"
srun -n1 -c7 --export=ALL \
  $(which python3) -W ignore -u ./phase4_run_trackers.py \
    --det-dir $BASE/nc --out-dir $BASE/tracks --years $YEARS
echo "### done"

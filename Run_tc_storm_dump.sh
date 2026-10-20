#!/bin/bash -l
# Run_tc_storm_dump.sh — per-storm, per-position diagnosis in the advisor's US box
# (phase4_tc_storm_dump.py), 1980-2024, 1 GPU node, one rank per year stride.
# Probabilities come from CKPT; 'e300s' tracks are built from the SAME checkpoint
# (Run_tc_export_track.sh NAME=e300s_hurr), so category D (tracker lost) is meaningful.
# Resumable: years with storms_Y.json are skipped.
#   sbatch Run_tc_storm_dump.sh
#SBATCH --account=CLI138
#SBATCH -J tc_dump
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:15:00
#SBATCH -o logs/tcdump-%j.txt
#SBATCH -e logs/tcdump-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7 MIOPEN_FIND_MODE=2 MIOPEN_DISABLE_CACHE=0
export IBTRACS_WIND_SOURCE=any
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
CKPT=${CKPT:-ARs_Hurricanes_ETCs_MCS/Phase2/H_final_e300_single/best_hurricane.pth}
OUTD=${OUTD:-phase4_results/storm_dump_usbox}
srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_tc_storm_dump.py --years $(seq 1980 2024) \
    --checkpoint "$CKPT" --outdir "$OUTD" \
    --tracks "shipped=$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2/Hurricanes_WNet_final" \
             "e300s=$NB/TC_candidates/e300s_hurr/tracks/hurricane"
echo "years done: $(ls $OUTD/storms_*.json 2>/dev/null | wc -l)/45  $(date)"

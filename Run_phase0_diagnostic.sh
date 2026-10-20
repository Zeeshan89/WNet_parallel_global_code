#!/bin/bash -l
# Run_phase0_diagnostic.sh — Phase 0 shortcut / input-ablation diagnostic.
#
# Single GPU, forward passes only, reads the precomputed 14-channel tensor cache
# so no ERA5 loading is needed. ~15 min for 4 seasonal months at stride 4.
#
# Submit:
#   sbatch Run_phase0_diagnostic.sh
#   sbatch --export=CHECKPOINT=/path/to.pth,MONTHS="201407 201401" Run_phase0_diagnostic.sh

#SBATCH --account=CLI138
#SBATCH -J wnet_phase0
#SBATCH -p batch
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
# Actual runtime is ~70 s. Keep the request tiny so the job backfills into the
# next open slot instead of waiting behind large reservations.
#SBATCH --time=00:10:00
#SBATCH -o logs/phase0-%j.txt
#SBATCH -e logs/phase0-err-%j.txt
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

export MIOPEN_FIND_MODE=2
export MIOPEN_DISABLE_CACHE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=7
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

CHECKPOINT="${CHECKPOINT:-/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Trail9/final_checkpoint_2006_2024.pth}"
# four seasonal months so the numbers are not a summer-only artefact
MONTHS="${MONTHS:-201401 201404 201407 201410}"
STRIDE="${STRIDE:-4}"
OUTDIR="${OUTDIR:-phase0_diagnostics}"

echo "================================================================================"
echo " Phase 0 shortcut diagnostic"
echo "   checkpoint : $CHECKPOINT"
echo "   months     : $MONTHS   (stride $STRIDE)"
echo "   outdir     : $OUTDIR"
echo "   started    : $(date)"
echo "================================================================================"

srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
    $mypython -W ignore -u ./phase0_shortcut_diagnostic.py \
        --checkpoint "$CHECKPOINT" \
        --months $MONTHS \
        --stride "$STRIDE" \
        --batch-size 4 \
        --outdir "$OUTDIR"

RC=$?
echo "Finished at $(date), exit $RC"
exit $RC

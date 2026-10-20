#!/bin/bash -l
# Run_export_class.sh — export one class from one checkpoint for a set of years (1 GPU node).
#   sbatch --export=ALL,CLASS=ETC,CKPT=...,OUT=...,YEARS="2000 2001" Run_export_class.sh
#SBATCH --account=CLI138
#SBATCH -J export_cls
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:15:00
#SBATCH -o logs/exportcls-%j.txt
#SBATCH -e logs/exportcls-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7 MIOPEN_FIND_MODE=2 MIOPEN_DISABLE_CACHE=0
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
: "${CLASS:?}" "${CKPT:?}" "${OUT:?}" "${YEARS:?}"
srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_export_detections.py --checkpoint "$CKPT" \
    --cache-dir "$NB/tensor_cache_v5_ibtracs" --years $YEARS --classes $CLASS --subdirs --outdir "$OUT"
echo "files: $(ls $OUT/*_WNet_final/*.nc 2>/dev/null | wc -l)  $(date)"

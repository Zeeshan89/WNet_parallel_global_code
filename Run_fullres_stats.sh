#!/bin/bash -l
# Run_fullres_stats.sh — predictor_stats_fullres.json from tensor_cache_v11_fullres (same method as predictor_stats.json).
#SBATCH --account=CLI138
#SBATCH -J fullres_stats
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=00:20:00
#SBATCH -o logs/fullresstats-%j.txt
#SBATCH -e logs/fullresstats-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH PYTHONNOUSERSITE=1
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
srun -n1 -c56 $(which python3) -u compute_predictor_stats.py --cache-dir $NB/tensor_cache_v11_fullres --moments \
  --out predictor_stats_fullres.json.tmp && mv predictor_stats_fullres.json.tmp predictor_stats_fullres.json
python3 -c "
import json; a=json.load(open('predictor_stats.json'))['stats']; b=json.load(open('predictor_stats_fullres.json'))['stats']
for k in a: print(f'{k:<11} 0.94deg mean {a[k][\"mean\"]:.4g} std {a[k][\"std\"]:.4g}   0.25deg mean {b[k][\"mean\"]:.4g} std {b[k][\"std\"]:.4g}')"

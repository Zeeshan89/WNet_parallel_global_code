#!/bin/bash -l
# Run_ar_objects_fullres.sh — AR object metrics (phase4_ar_objects_v2.py + merge_ar_objects_v2.py) for the 0.25 deg
# AR product into evaluations/ar_mars_eval_fullres, as for v2.
#SBATCH --account=CLI138
#SBATCH -J ar_obj_fr
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH -o logs/arobj-fr-%j.txt
#SBATCH -e logs/arobj-fr-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
OUT=$NB/WNet_parallel_global_code/evaluations/ar_mars_eval_fullres
srun -n 32 -c 1 --export=ALL $(which python3) -W ignore -u phase4_ar_objects_v2.py \
  --pred-dir $NB/Global_ARs_Hurr_ETCs_MCS_fullres025/ARs_WNet_final --outdir $OUT | tail -3
cd $OUT && $(which python3) -W ignore -u ../ar_mars_eval_v2/merge_ar_objects_v2.py --cache-dir . --label fullres | tail -8

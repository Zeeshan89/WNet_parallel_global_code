#!/bin/bash -l
# Run_repair_h5_201309.sh — regenerate the corrupt preprocessed_v3/era5_preprocessed_201309.h5 ("filter returned
# failure during read"; mslp/ivt/ivt_grad) with repair_preprocessed_h5.py into a staging dir, check it, swap it in (corrupt copy kept),
# then build the 0.25 deg cache month 201309 (removing the empty placeholder first).
#SBATCH --account=CLI138
#SBATCH -J repair_201309
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=00:30:00
#SBATCH -o logs/repair201309-%j.txt
#SBATCH -e logs/repair201309-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH PYTHONNOUSERSITE=1 HDF5_USE_FILE_LOCKING=FALSE
PY=$(which python3)
H5DIR=../Percentile_ARs_detection/cache/preprocessed_v3
STAGE=../Percentile_ARs_detection/cache/preprocessed_v3_repair
mkdir -p $STAGE
set -e
srun -n1 -c56 $PY -u repair_preprocessed_h5.py --tag 201309 --out $STAGE/era5_preprocessed_201309.h5
mv $H5DIR/era5_preprocessed_201309.h5 $H5DIR/era5_preprocessed_201309.h5.corrupt
cp $STAGE/era5_preprocessed_201309.h5 $H5DIR/era5_preprocessed_201309.h5
C=../tensor_cache_v11_fullres
[ -s $C/split_201309.npy ] || rm -f $C/split_201309.npy
export OMP_NUM_THREADS=1 FULLRES_NPROC=56
srun -n1 -c56 $PY -W ignore -u build_fullres_cache.py --years 2013 2013 --months 9 | grep -E "\] |PASS|FAIL|ERROR"
ls -la $C/split_201309.npy

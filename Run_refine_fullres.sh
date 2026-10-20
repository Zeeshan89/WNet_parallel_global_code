#!/bin/bash -l
# Run_refine_fullres.sh — 0.25 deg shape refinement of the v2 AR / MCS masks (refine_fullres.py).
# Writes to Refined_fullres_candidates/<NAME>; the shared _v2 products are not touched.
# Resumable per month.  sbatch --export=ALL,KIND=ar,SRC=...,NAME=ARs_6h Run_refine_fullres.sh
#SBATCH --account=CLI138
#SBATCH -J refine
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:15:00
#SBATCH -o logs/refine-%j.txt
#SBATCH -e logs/refine-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
PY=/ccs/home/zaslam/.conda/envs/my_env/bin/python
export OMP_NUM_THREADS=4 PYTHONNOUSERSITE=1 HDF5_USE_FILE_LOCKING=FALSE
: "${KIND:?}" "${SRC:?}" "${NAME:?}"
YEARS="${YEARS:-$(seq 1980 2024 | tr '\n' ' ')}"
DST=$NB/Refined_fullres_candidates/$NAME
srun -n $SLURM_NTASKS -c 7 --export=ALL $PY -u refine_fullres.py --kind $KIND --grow ${GROW:-1} \
    --src "$SRC" --dst "$DST" --years $YEARS > logs/refine_${NAME}_${SLURM_JOB_ID}.log 2>&1
N=$(ls $DST/*.nc 2>/dev/null | wc -l); echo "$NAME: $N months written $(date)"

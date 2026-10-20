#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J learnchk
#SBATCH -p batch
#SBATCH --nodes=3
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH -o logs/learnchk-%j.txt
#SBATCH -e logs/learnchk-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7
export YEARS="${YEARS:-2003 2012}"
# Two years, all months: confirms the 0.996 signature rate is not a peak-Atlantic
# artefact, and covers the DJF S-hemisphere season where the spurious tracks live.
srun -n $SLURM_NTASKS -c 7 --export=ALL bash -c '
  set -- $YEARS; yi=$((SLURM_PROCID/12)); m=$((SLURM_PROCID%12+1))
  [ "$yi" -ge "$#" ] && exit 0
  eval y=\${$((yi+1))}
  python3 -W ignore -u ./phase4_ibtracs_learnability.py --year $y --month $m \
      > "logs/learn_${y}${m}.log" 2>&1
  echo "[rank $SLURM_PROCID] $y-$m rc=$?"
'
echo "done $(date); files: $(ls tc_learnability/*.json 2>/dev/null | wc -l)"

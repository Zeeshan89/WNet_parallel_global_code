#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J tc_tune
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=11
#SBATCH --time=00:40:00
#SBATCH -o logs/tctune-%j.txt
#SBATCH -e logs/tctune-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=11
export BASE=ARs_Hurricanes_ETCs_MCS/Phase2_E_ibtracs
export DET="$BASE/nc"
export YEARS="${YEARS:-1983 1994 2003 2012 2021}"
# Variant A is the current default and doubles as the rejection diagnostic.
# B-D progressively relax the gates that were tuned for a 24x noisier detector.
srun -n $SLURM_NTASKS -c 11 --export=ALL bash -c '
  set -- $YEARS; i=$((SLURM_PROCID+1)); [ "$i" -gt "$#" ] && exit 0; eval y=\${$i}
  for V in A B C D; do
    case $V in
      A) unset TC_MIN_POINTS TC_MIN_DURATION_H TC_MIN_PATH_KM TC_MIN_INTENSITY TC_MIN_STRAIGHTNESS TC_MIN_OCEAN_FRAC ;;
      B) export TC_MIN_DURATION_H=24 TC_MIN_PATH_KM=150 TC_MIN_POINTS=4 ;;
      C) export TC_MIN_DURATION_H=24 TC_MIN_PATH_KM=150 TC_MIN_POINTS=4 TC_MIN_INTENSITY=0.30 ;;
      D) export TC_MIN_DURATION_H=18 TC_MIN_PATH_KM=100 TC_MIN_POINTS=3 TC_MIN_INTENSITY=0.25 TC_MIN_STRAIGHTNESS=0.05 TC_MIN_OCEAN_FRAC=0.20 ;;
    esac
    O="$BASE/tune_$V"
    [ -s "$O/hurricane/CONUS_year_${y}/CONUS_hurricane_tracks_detailed_${y}.txt" ] && continue
    python3 -W ignore -u ./phase4_run_trackers.py --det-dir "$DET" --out-dir "$O" \
        --years $y --which hurricane > "logs/tctune_${V}_${y}.log" 2>&1
    echo "[rank $SLURM_PROCID] $V $y rc=$?"
  done
'
echo "done $(date)"

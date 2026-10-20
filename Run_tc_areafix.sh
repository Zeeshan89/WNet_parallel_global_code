#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J tc_areafix
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=11
#SBATCH --time=00:30:00
#SBATCH -o logs/tcarea-%j.txt
#SBATCH -e logs/tcarea-err-%j.txt
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
# E: area cap only, isolating the fix (measured: old cap drops 34.1% of blobs).
# F: area cap + the mild duration/path relaxations, to see if they add anything
#    once the real bottleneck is gone.
srun -n $SLURM_NTASKS -c 11 --export=ALL bash -c '
  set -- $YEARS; i=$((SLURM_PROCID+1)); [ "$i" -gt "$#" ] && exit 0; eval y=\${$i}
  for V in E F; do
    case $V in
      E) export TC_MAX_AREA_PX=2000; unset TC_MIN_DURATION_H TC_MIN_PATH_KM TC_MIN_POINTS ;;
      F) export TC_MAX_AREA_PX=2000 TC_MIN_DURATION_H=24 TC_MIN_PATH_KM=150 TC_MIN_POINTS=4 ;;
    esac
    O="$BASE/tune_$V"
    [ -s "$O/hurricane/CONUS_year_${y}/CONUS_hurricane_tracks_detailed_${y}.txt" ] && continue
    python3 -W ignore -u ./phase4_run_trackers.py --det-dir "$DET" --out-dir "$O" \
        --years $y --which hurricane > "logs/tctune_${V}_${y}.log" 2>&1
    echo "[rank $SLURM_PROCID] $V $y rc=$?"
  done
'
echo "done $(date)"

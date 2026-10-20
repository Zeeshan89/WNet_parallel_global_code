#!/bin/bash -l
# Run_tc_tracks_fixhav.sh — re-track the v2 TC masks after the haversine fix in
# Hurricane_track.py:is_realistic_hurricane_movement. The old code swapped lat and
# lon, which cut tracks at landfalls near 90W and 90E.
#
# Pass A (legacy): TC_HAVERSINE_LEGACY=1 on a few years; must reproduce the shipped
#                  v2 .txt byte-for-byte, so every later difference is the fix alone.
# Pass B (fixed) : all years, fixed code, same masks, same env as the v2 product.
#
# Resumable: a year whose detailed .txt exists is skipped.
#   sbatch -q debug Run_tc_tracks_fixhav.sh
#SBATCH --account=CLI138
#SBATCH -J tc_fixhav
#SBATCH -p batch
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH -o logs/tcfixhav-%j.txt
#SBATCH -e logs/tcfixhav-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7
export TC_MAX_AREA_PX=2000 IBTRACS_WIND_SOURCE=any TC_MIN_DURATION_H=24

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export DET=${DET:-$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2/Hurricanes_WNet_final}
export REF=$DET                      # shipped v2 tracks live next to the masks
export OUTL=${OUTL:-ARs_Hurricanes_ETCs_MCS/Phase2_E_ibtracs/tracks_v2_legacy_check}
export OUTF=${OUTF:-ARs_Hurricanes_ETCs_MCS/Phase2_E_ibtracs/tracks_fixhav}
export LEGACY_YEARS="${LEGACY_YEARS:-1983 1986 2005}"
export Y0=${Y0:-1980}; export Y1=${Y1:-2024}

# One job list: legacy years first, then all fixed years; strided over ranks.
JOBS=""
for y in $LEGACY_YEARS; do JOBS="$JOBS L:$y"; done
for y in $(seq $Y0 $Y1); do JOBS="$JOBS F:$y"; done
export JOBS
srun -n $SLURM_NTASKS -c 7 --export=ALL bash -c '
  set -- $JOBS; i=0
  for j in "$@"; do
    if [ $((i % SLURM_NTASKS)) -eq $SLURM_PROCID ]; then
      mode=${j%%:*}; y=${j##*:}
      if [ "$mode" = L ]; then out=$OUTL; leg=1; else out=$OUTF; leg=0; fi
      f="$out/hurricane/CONUS_year_${y}/CONUS_hurricane_tracks_detailed_${y}.txt"
      if [ -s "$f" ]; then echo "[$j] already done"; else
        TC_HAVERSINE_LEGACY=$leg python3 -W ignore -u ./phase4_run_trackers.py \
            --det-dir "$DET" --out-dir "$out" --years $y --which hurricane \
            > "logs/tcfixhav_${mode}${y}.log" 2>&1
        echo "[rank $SLURM_PROCID] $j rc=$?"
      fi
    fi
    i=$((i + 1))
  done
'
echo "=== legacy byte check vs shipped v2 ==="
for y in $LEGACY_YEARS; do
  a="$OUTL/hurricane/CONUS_year_${y}/CONUS_hurricane_tracks_detailed_${y}.txt"
  b="$REF/CONUS_year_${y}/CONUS_hurricane_tracks_detailed_${y}.txt"
  if cmp -s "$a" "$b"; then echo "  $y IDENTICAL"; else echo "  $y DIFFERS"; fi
done
NDONE=$(ls $OUTF/hurricane/CONUS_year_*/CONUS_hurricane_tracks_detailed_*.txt 2>/dev/null | wc -l)
echo "fixed years with tracks: $NDONE/$((Y1-Y0+1))"
echo "done $(date)"

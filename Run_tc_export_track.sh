#!/bin/bash -l
# Run_tc_export_track.sh — one TC candidate end to end in a single 2-node job:
#   1) export the Hurricane mask from CKPT (GPU, 16 ranks)
#   2) track it with the FIXED tracker (haversine fix), production env
# into $NB/TC_candidates/$NAME/{Hurricanes_WNet_final, tracks/hurricane/CONUS_year_Y}.
# Both stages skip what already exists, so a walltime kill is fixed by resubmitting.
# Score afterwards with hurricane_analysis/tc_track_compare.py.
#
#   sbatch --export=ALL,NAME=e300s_hurr,CKPT=ARs_Hurricanes_ETCs_MCS/Phase2/H_final_e300_single/best_hurricane.pth Run_tc_export_track.sh
#   sbatch --export=ALL,NAME=e300s_hurr_t040,THR=0.4,YEARS="1983 1987 ...",CKPT=... Run_tc_export_track.sh
#SBATCH --account=CLI138
#SBATCH -J tc_cand
#SBATCH -p batch
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:25:00
#SBATCH -o logs/tccand-%j.txt
#SBATCH -e logs/tccand-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7
export MIOPEN_FIND_MODE=2 MIOPEN_DISABLE_CACHE=0

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
: "${NAME:?set NAME}"; : "${CKPT:?set CKPT}"
THR="${THR:-0.5}"
YEARS="${YEARS:-$(seq 1980 2024 | tr '\n' ' ')}"
export BASE="$NB/TC_candidates/$NAME"
export DET="$BASE/Hurricanes_WNet_final" OUTD="$BASE/tracks"
[ -s "$CKPT" ] || { echo "ERROR: $CKPT missing"; exit 1; }
mkdir -p "$BASE"; NY=$(echo $YEARS | wc -w)
echo "candidate $NAME  ckpt=$CKPT  thr=$THR  years=$NY  $(date)"

echo "=== 1) export ==="
srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_export_detections.py \
    --checkpoint "$CKPT" --cache-dir "$NB/tensor_cache_v5_ibtracs" \
    --years $YEARS --classes Hurricane --subdirs --outdir "$BASE" --prob-threshold "$THR"
NF=$(ls $DET/*.nc 2>/dev/null | wc -l); echo "exported $NF / $((NY * 12)) months"
[ "$NF" -ge $((NY * 12)) ] || { echo "INCOMPLETE export — resubmit"; exit 1; }

echo "=== 2) track (fixed tracker) ==="
export TC_MAX_AREA_PX=2000 IBTRACS_WIND_SOURCE=any TC_MIN_DURATION_H=24 YEARS
srun -n $SLURM_NTASKS -c 7 --export=ALL bash -c '
  i=0
  for y in $YEARS; do
    if [ $((i % SLURM_NTASKS)) -eq $SLURM_PROCID ]; then
      f="$OUTD/hurricane/CONUS_year_${y}/CONUS_hurricane_tracks_detailed_${y}.txt"
      [ -s "$f" ] || { python3 -W ignore -u ./phase4_run_trackers.py --det-dir "$DET" \
          --out-dir "$OUTD" --years $y --which hurricane > "logs/tccand_${SLURM_JOB_ID}_${y}.log" 2>&1
          echo "[rank $SLURM_PROCID] $y rc=$?"; }
    fi
    i=$((i + 1))
  done
'
NT=$(ls $OUTD/hurricane/CONUS_year_*/CONUS_hurricane_tracks_detailed_*.txt 2>/dev/null | wc -l)
echo "tracked years: $NT / $NY   done $(date)"

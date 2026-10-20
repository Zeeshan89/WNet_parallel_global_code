#!/bin/bash -l
# Run_build_hourly_ALL_andes.sh — build the FULL 1980-2024 hourly BT + moisture in ONE
# multi-node Andes job (Andes allows MaxNodes/MaxTime UNLIMITED; only MaxJobs=1).
# 45 years are distributed round-robin across NODES (one worker task per node, 32 threads).
# Each worker builds, for each of its years:
#   BT       -> OLR/brightness_temp_hourly/era5_bt_hourly_{YEAR}.nc   (12 months)
#   moisture -> data_hourly/era5dara_hourly_{YEAR}{MM}.nc             (IWV=TCWV + zeros)
#
# Submit: sbatch -M andes -D <codedir> Run_build_hourly_ALL_andes.sh

#SBATCH -A cli138
#SBATCH -J bt_moist_ALL
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 9
#SBATCH --ntasks=9
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 08:00:00
#SBATCH -o logs/buildALL-%j.txt
#SBATCH -e logs/buildALL-err-%j.txt

set -u
export CODE=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code
export PY=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/bin/python
export BTDIR=/lustre/orion/proj-shared/cli138/z7a/Storms_data/OLR/brightness_temp_hourly
export DATAH=/lustre/orion/cli138/proj-shared/z7a/Storms_data/data_hourly
export Y0=1980 Y1=2024
cd "$CODE"; mkdir -p logs "$BTDIR" "$DATAH"

echo "=== BUILD ALL hourly inputs ${Y0}-${Y1} on $SLURM_NNODES nodes  start=$(date) ==="

srun -N "$SLURM_NNODES" -n "$SLURM_NTASKS" --ntasks-per-node=1 -c 32 --export=ALL bash -c '
  rank=$SLURM_PROCID; world=$SLURM_NTASKS
  export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
  export LD_LIBRARY_PATH=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/lib:${LD_LIBRARY_PATH:-}
  export PYTHONNOUSERSITE=1
  for y in $(seq $((Y0+rank)) $world $Y1); do
    echo "[rank $rank/$world] === YEAR $y start $(date) ==="
    bt="$BTDIR/era5_bt_hourly_${y}.nc"
    # skip only a COMPLETE year (~37 GB); a partial/July-only file (~3 GB) is rebuilt
    sz=$(stat -c%s "$bt" 2>/dev/null || echo 0)
    if [ "$sz" -gt 30000000000 ]; then echo "[rank $rank] BT $y complete (${sz}B), skip"; else
      "$PY" "$CODE/build_hourly_bt.py" --year $y --months 1,2,3,4,5,6,7,8,9,10,11,12 --out "$bt" \
        || echo "[rank $rank] !! BT $y FAILED"
    fi
    for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
      out="$DATAH/era5dara_hourly_${y}${m}.nc"
      if [ -s "$out" ]; then continue; fi
      "$PY" "$CODE/build_hourly_moisture.py" --year $y --month ${m#0} --out "$out" \
        || echo "[rank $rank] !! moist ${y}${m} FAILED"
    done
    echo "[rank $rank/$world] === YEAR $y DONE $(date)  BT=$(du -h "$bt" 2>/dev/null|cut -f1) moist=$(ls $DATAH/era5dara_hourly_${y}??.nc 2>/dev/null|wc -l)/12 ==="
  done
  echo "[rank $rank/$world] ALL ASSIGNED YEARS DONE $(date)"
'
echo "=== BUILD ALL finished  $(date) ==="
echo "BT files:   $(ls $BTDIR/era5_bt_hourly_*.nc 2>/dev/null | wc -l)"
echo "moist files:$(ls $DATAH/era5dara_hourly_*.nc 2>/dev/null | wc -l) (expect ~540)"

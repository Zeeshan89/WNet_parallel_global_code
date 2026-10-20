#!/bin/bash -l
# Run_build_hourly_year_andes.sh — build one YEAR of hourly inputs on Andes CPU:
#   BT       -> OLR/brightness_temp_hourly/era5_bt_hourly_{YEAR}.nc   (12 months, valid_time)
#   moisture -> data_hourly/era5dara_hourly_{YEAR}{MM}.nc             (IWV=TCWV + zeros)
# Submit: sbatch -M andes -D <codedir> --export=YEAR=2005 Run_build_hourly_year_andes.sh

#SBATCH -A cli138
#SBATCH -J bt_moist_build
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 03:00:00
#SBATCH -o logs/build-%j.txt
#SBATCH -e logs/build-err-%j.txt

set -u
: "${YEAR:?set YEAR via --export=YEAR=YYYY}"
CODE=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code
PY=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/bin/python
BTDIR=/lustre/orion/proj-shared/cli138/z7a/Storms_data/OLR/brightness_temp_hourly
DATAH=/lustre/orion/cli138/proj-shared/z7a/Storms_data/data_hourly
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$CODE"; mkdir -p logs "$BTDIR" "$DATAH"

echo "=== BUILD hourly inputs YEAR=$YEAR  start=$(date) ==="

echo ">>> [1/2] hourly BT (12 months) -> era5_bt_hourly_${YEAR}.nc"
$PY build_hourly_bt.py --year "$YEAR" --months 1,2,3,4,5,6,7,8,9,10,11,12 \
    --out "$BTDIR/era5_bt_hourly_${YEAR}.nc"
rc1=$?; echo ">>> BT rc=$rc1  $(date)"

echo ">>> [2/2] hourly moisture (12 months) -> era5dara_hourly_${YEAR}MM.nc"
rc2=0
for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    $PY build_hourly_moisture.py --year "$YEAR" --month "${m#0}" \
        --out "$DATAH/era5dara_hourly_${YEAR}${m}.nc" || rc2=1
done
echo ">>> moisture rc=$rc2  $(date)"

echo "=== DONE YEAR=$YEAR  BT_rc=$rc1 moist_rc=$rc2  $(date) ==="
ls -la "$BTDIR/era5_bt_hourly_${YEAR}.nc" 2>/dev/null | awk '{printf "  BT %.1f GB\n",$5/1e9}'
echo "  moisture files: $(ls $DATAH/era5dara_hourly_${YEAR}??.nc 2>/dev/null | wc -l)/12"
exit $(( rc1 || rc2 ))

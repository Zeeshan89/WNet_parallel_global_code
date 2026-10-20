#!/bin/bash -l
# Run_export_fullres.sh — 6-hourly detections from the 0.25 deg WNet (J_fullres_e60), run ON the 0.25 deg
# grid (phase4_export_detections.py --upsample native: tensor_cache_v11_fullres predictors, rows padded
# 721 -> 728, output cropped back; no resampling). One class per pass, each from its own checkpoint:
#   AR <- best_ar.pth (AR IoU)            Hurricane <- best_hurricane.pth (TC recall at P >= 0.60)
#   ETC <- best_etc.pth (ETC IoU)          MCS <- best_mcs.pth (MCS IoU)
# Output: Global_ARs_Hurr_ETCs_MCS_fullres025/{ARs,Hurricanes,ETCs,MCS}_WNet_final (v2 untouched).
# 3 ranks share each GPU (the NetCDF write is CPU-bound). Resumable: existing months are skipped.
#   sbatch Run_export_fullres.sh                              # all classes, 1980-2024
#   sbatch --export=ALL,CLASSES="AR MCS",YEARS="2003" Run_export_fullres.sh
#SBATCH --account=CLI138
#SBATCH -J exp_fullres
#SBATCH -p batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH -o logs/expfullres-%j.txt
#SBATCH -e logs/expfullres-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MIOPEN_FIND_MODE=2 MIOPEN_DISABLE_CACHE=0
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
CK=ARs_Hurricanes_ETCs_MCS/Phase2/${TAG:-J_fullres_e60}
OUT="${OUT:-$NB/Global_ARs_Hurr_ETCs_MCS_fullres025}"
CLASSES="${CLASSES:-AR Hurricane ETC MCS}"
YEARS="${YEARS:-$(seq 1980 2024 | tr '\n' ' ')}"
declare -A CKPT=([AR]=best_ar.pth [Hurricane]=best_hurricane.pth [ETC]=best_etc.pth [MCS]=best_mcs.pth)
N=$((SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE))
echo "0.25 deg export: classes [$CLASSES] -> $OUT  ranks=$N  $(date)"
for c in $CLASSES; do
  ck=$CK/${CKPT[$c]}
  [ -s "$ck" ] || { echo "ERROR: $ck missing"; exit 1; }
  echo "=== $c <- $ck (epoch $(python3 -c "import torch;print(torch.load('$ck',map_location='cpu',weights_only=False).get('epoch'))" 2>/dev/null))"
  srun -N $SLURM_JOB_NUM_NODES -n $N -c 2 --gpus-per-node=8 --export=ALL \
    $(which python3) -W ignore -u ./phase4_export_detections.py --checkpoint "$ck" --classes $c \
      --cache-dir "$NB/tensor_cache_v11_fullres" --stats predictor_stats_fullres.json \
      --years $YEARS --subdirs --outdir "$OUT" --upsample native --batch-size ${BS:-2} | grep -v "steps ->"
done
for d in ARs Hurricanes ETCs MCS; do echo "  ${d}_WNet_final: $(ls $OUT/${d}_WNet_final 2>/dev/null | wc -l)/540"; done
echo "done $(date)"

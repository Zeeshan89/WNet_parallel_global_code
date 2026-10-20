#!/bin/bash -l
# Run_regcm_tc_zeroshot.sh — TC detections for ONE RegCM model, by running the
# ERA5 IBTrACS-trained TC head on the RegCM predictors. No RegCM training.
#
#   sbatch Run_regcm_tc_zeroshot.sh NOR
#
# WHY ZERO-SHOT RATHER THAN FINE-TUNE
# -----------------------------------
# Both available RegCM TC teachers are weaker than the head we already have:
#   TempestExtremes  scored CSI 0.388 on ERA5 against this head's 0.708, and its
#                    StitchNodes box covers only 31.8% of the RegCM domain
#   physics detector supervises the other 68.2%; the equivalent CyTRACK detector
#                    on ERA5 had a false-alarm floor of 0.909
# This project's repeated finding is that the label is the ceiling, so fine-tuning
# the TC head on either would pull it DOWN. The ERA5 head's teacher is IBTrACS
# observations, and it applies uniformly over the WHOLE domain -- no dependence on
# TE's search box, no provenance split.
#
# WHY ERA5 NORMALISATION STATS, NOT RegCM's
# -----------------------------------------
# A 980 hPa low must map to the same network input in both datasets; normalising
# RegCM by its own sigma would rescale the physics. Measured pixel-wise inside the
# domain mask, the TC-relevant predictors agree closely with ERA5 anyway:
#   MSLP  mean +0.59 sigma, std ratio 0.60      vorticity  +0.02 sigma, 0.88
#   wind  +0.02..0.06 sigma, 0.83-0.88          (BT is 2.31x, but BT feeds MCS)
# The old "sigmas differ 2-25x" blocker was an artifact of comparing global against
# regional FIELD-MEANS; computed per pixel it does not hold.
# CAVEAT: MSLP std ratio 0.60 means RegCM lows look ~40% weaker in standardised
# units, biasing toward UNDER-detection. If detections come out sparse, that is the
# first suspect and a per-predictor rescale is the lever.
#
# GRID: exports on the RegCM NATIVE 156x328 grid with ASCENDING latitude (18.00 ..
# 56.75 N, -138.75 .. -57.00 E), which is the grid the <MODEL>dara_ met fields live
# on and the one Hurricane_track.py derives its domain from. --out-mask is
# MANDATORY: training used a valid mask, so the model is unconstrained outside the
# domain and emits arbitrary output there (measured previously: 84% of NOR and 89%
# of MPI detections fell outside without it).

#SBATCH --account=CLI138
#SBATCH -J regcm_tc_zs
#SBATCH -p batch
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:20:00
#SBATCH -o logs/regcmtc-%j.txt
#SBATCH -e logs/regcmtc-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV

MODEL="${1:?usage: sbatch Run_regcm_tc_zeroshot.sh <ACC|BCC|CNR|MPI|MRI|NOR>}"
case "$MODEL" in ACC|BCC|CNR|MPI|MRI|NOR) ;; *) echo "ERROR: bad model '$MODEL'"; exit 1;; esac

module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7
export MIOPEN_FIND_MODE=2 MIOPEN_DISABLE_CACHE=0

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
CACHE=$NB/tensor_cache_RegCM_split_$MODEL
CKPT=ARs_Hurricanes_ETCs_MCS/Phase2/F_ibtracs_pwH50/best_hurricane.pth
STATS=predictor_stats.json
OUT=${OUT:-$NB/RegCM_TC_zeroshot/$MODEL}
YEARS=$(seq 1980 2014 | tr '\n' ' ')

for f in "$CKPT" "$STATS" "$CACHE/native_lat.npy" "$CACHE/native_lon.npy" \
         "$CACHE/native_domain_eroded.npy"; do
  [ -s "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

echo "model      : $MODEL"
echo "checkpoint : $CKPT   (ERA5 IBTrACS-trained TC head, epoch 13, recall 0.754)"
echo "stats      : $STATS  (ERA5, deliberately)"
echo "cache      : $CACHE"
echo "outdir     : $OUT"
echo "months     : $(ls $CACHE/split_*.npy 2>/dev/null | grep -vc meta)"
echo "started    : $(date)"

srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_export_detections.py \
    --checkpoint "$CKPT" --cache-dir "$CACHE" --stats "$STATS" \
    --years $YEARS --classes Hurricane --subdirs \
    --out-lat "$CACHE/native_lat.npy" --out-lon "$CACHE/native_lon.npy" \
    --out-mask "$CACHE/native_domain_eroded.npy" \
    --file-prefix "${MODEL}dara_" --outdir "$OUT"
RC=$?
echo "exit $RC"
echo "files: $(ls $OUT/Hurricanes_WNet_final 2>/dev/null | wc -l)  (expect 420)"
du -sh "$OUT" 2>/dev/null
echo "Finished $(date)"
exit $RC

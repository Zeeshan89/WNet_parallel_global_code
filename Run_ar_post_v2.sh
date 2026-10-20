#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J ar_post
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH -o logs/arpost-%j.txt
#SBATCH -e logs/arpost-err-%j.txt
#SBATCH --export=NONE
#
# Verify, merge and plot the AR evaluation. This runs on a compute node rather
# than the login node because merging the 32 part files means decompressing
# roughly 15 GB of npz, which the shared login node will not do in reasonable
# time.
#
# Stages are guarded so the job is useful before every input exists:
#   AR_STAGE=v2    verify + merge v2, plus the figures that need only v2
#   AR_STAGE=all   the above, plus the v1 merge, the 3-row figure, diagnostics
#                  and the object-metrics merge
#
#   sbatch Run_ar_post_v2.sh
#   sbatch --export=ALL,AR_STAGE=all Run_ar_post_v2.sh

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=32
export MPLBACKEND=Agg

STAGE=${AR_STAGE:-v2}
CODE=/lustre/orion/cli138/proj-shared/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code
V2=$CODE/evaluations/ar_mars_eval_v2
V1=$CODE/evaluations/ar_mars_eval_v1_recomputed
PY=$(which python3)
cd $V2

echo "############ stage=$STAGE ############"
echo "############ verify v2 ############"
$PY -u verify_ar_v2.py --cache-dir $V2
VERIFY=$?
echo "verify exit $VERIFY"

echo "############ merge v2 ############"
$PY -u merge_ar_v2.py --cache-dir $V2 --label v2 || exit 1

echo "############ seasonal figure ############"
$PY -u plot_ar_v2_seasonal.py --v2-dir $V2

echo "############ timeseries figure ############"
$PY -u plot_ar_v2_timeseries.py --v2-dir $V2

if [ "$STAGE" = "all" ]; then
  if ls $V1/part_*.npz >/dev/null 2>&1; then
    echo "############ merge v1 ############"
    $PY -u merge_ar_v2.py --cache-dir $V1 --label v1 \
        --pred-name "Global_ARs_Hurr_ETCs_MCS_finalized/ARs_WNet_final"
  else
    echo "no v1 parts in $V1, skipping v1 merge"
  fi
  echo "############ main 3-row figure ############"
  $PY -u plot_ar_v2_main.py --v2-dir $V2 --v1-dir $V1
  echo "############ diagnostics figure ############"
  $PY -u plot_ar_v2_diagnostics.py --v2-dir $V2 --v1-dir $V1
  if ls $V2/objpart_*.json >/dev/null 2>&1; then
    echo "############ object metrics ############"
    $PY -u merge_ar_objects_v2.py --cache-dir $V2 --label v2
  else
    echo "no objpart_*.json yet, skipping object merge"
  fi
fi

echo "############ outputs ############"
ls -la $V2/*.png $V2/*.pdf $V2/*.csv $V2/*.txt $V2/*.json 2>/dev/null
echo "done (verify exit $VERIFY)"

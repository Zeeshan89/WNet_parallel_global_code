#!/bin/bash
RANK=${SLURM_PROCID}
MY_CACHE="/tmp/miopen-cache-rank-${RANK}"
mkdir -p ${MY_CACHE}

LUSTRE_CACHE="${MIOPEN_LUSTRE_CACHE}"
if [[ -d "${LUSTRE_CACHE}" ]] \
   && [[ -n "$(ls -A ${LUSTRE_CACHE} 2>/dev/null)" ]] \
   && [[ -z "$(ls -A ${MY_CACHE} 2>/dev/null)" ]]; then
    cp -r ${LUSTRE_CACHE}/* ${MY_CACHE}/ 2>/dev/null || true
    sync
fi

echo "Rank ${RANK}: MIOpen cache ready — $(ls ${MY_CACHE} 2>/dev/null | wc -l) files" >&2

export MIOPEN_USER_DB_PATH="${MY_CACHE}"
export MIOPEN_CUSTOM_CACHE_DIR="${MY_CACHE}"
export MIOPEN_FIND_MODE=2
export MIOPEN_DISABLE_CACHE=0

# Disable slow solvers — prevents 863-run search on Find-Db miss
# These are the solvers causing the exhaustive SearchImpl loops.
# Remove these lines after re-running warmup with correct batch size.
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
export MIOPEN_DEBUG_CONV_FFT=0
export MIOPEN_DEBUG_CONV_WINOGRAD=0
export MIOPEN_DEBUG_CONV_DIRECT=1

exec "$@"

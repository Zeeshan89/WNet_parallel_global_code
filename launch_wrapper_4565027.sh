#!/bin/bash
# Per-rank MIOpen cache isolation.
# Copies Lustre Find-Db to node-local /tmp before MIOpen initializes,
# avoiding SQLite lock contention across 32 simultaneous ranks.

RANK=${SLURM_PROCID}
MY_CACHE="/tmp/miopen-cache-rank-${RANK}"
mkdir -p ${MY_CACHE}

# Seed from Lustre cache (only if local dir is empty — skip on re-use)
LUSTRE_CACHE="${MIOPEN_LUSTRE_CACHE}"
if [[ -d "${LUSTRE_CACHE}" ]] \
   && [[ -n "$(ls -A ${LUSTRE_CACHE} 2>/dev/null)" ]] \
   && [[ -z "$(ls -A ${MY_CACHE} 2>/dev/null)" ]]; then
    cp -r ${LUSTRE_CACHE}/* ${MY_CACHE}/ 2>/dev/null || true
    sync
fi

# Confirm cache is ready (visible in error log for debugging)
FILE_COUNT=$(ls ${MY_CACHE} 2>/dev/null | wc -l)
echo "Rank ${RANK}: MIOpen cache seeded — ${FILE_COUNT} files in ${MY_CACHE}" >&2

if [[ ${FILE_COUNT} -eq 0 ]]; then
    echo "Rank ${RANK}: WARNING — cache is empty, MIOpen will search kernels!" >&2
fi

# Point MIOpen to this rank's isolated local cache
# FAST (2): Find-Db hit = instant; miss = Immediate (no long search)
# DO NOT set MIOPEN_FIND_ENFORCE or MIOPEN_SYSTEM_DB_PATH (stoul crash)
export MIOPEN_USER_DB_PATH="${MY_CACHE}"
export MIOPEN_CUSTOM_CACHE_DIR="${MY_CACHE}"
export MIOPEN_FIND_MODE=3    
export MIOPEN_DISABLE_CACHE=0


exec "$@"

#!/usr/bin/env bash

set -Eeuo pipefail
if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 SUITE CONDITION GPU_ID" >&2
    exit 2
fi

SUITE="$1"
[[ -z "${LIBERO_VPRO_SUITE:-}" || "$SUITE" == "$LIBERO_VPRO_SUITE" ]] || exit 0
CONDITION="$2"
GPU_ID="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_EVAL_ROOT="${D1_MODEL_EVAL_ROOT:-${HERE}/../../outputs/D1}"
TRIAL_START="${LIBERO_VPRO_TRIAL_START:-${D1_TRIAL_START:-0}}"
TRIAL_COUNT="${LIBERO_VPRO_TRIALS:-${D1_TRIAL_COUNT:-10}}"
TASK_IDS_RAW="${LIBERO_VPRO_TASK:-${D1_TASK_IDS:-}}"
RUN_TAG="${D1_RUN_TAG:-full}"
RUN_ROOT="${MODEL_EVAL_ROOT}/fastwam/${SUITE}/${CONDITION}"
STATUS="${RUN_ROOT}/status.tsv"

case "${SUITE}" in
    libero_object|libero_spatial|libero_goal|libero_10) ;;
    *)
        echo "Unsupported suite: ${SUITE}" >&2
        exit 2
        ;;
esac

source "$HERE/../../scripts/fastwam_env.sh"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
export WAM_RECORD_EPISODES=1
export WAM_RECORD_PROFILE=d1_compact
export WAM_RECORD_COMPRESS=1
export WAM_EPISODE_RECORD_ROOT="${MODEL_EVAL_ROOT}/episode_records"
export WAM_TRACK=D1
export WAM_SUBTASK=D1
export WAM_CASE_ID="${CONDITION}"
export WAM_CONDITION=perturbed
unset FASTWAM_OBS_OCCLUSION_SPEC LIBERO_OBS_OCCLUSION_SPEC
unset FASTWAM_DECISION_SCHEDULE_SPEC LIBERO_SCHEDULE_SPEC

mkdir -p "${RUN_ROOT}"
touch "${STATUS}"
if [[ -n "${TASK_IDS_RAW}" ]]; then
    read -r -a TASK_IDS <<< "${TASK_IDS_RAW}"
else
    TASK_IDS=($(seq 0 9))
fi
for TASK_ID in "${TASK_IDS[@]}"; do
    if [[ ! "${TASK_ID}" =~ ^[0-9]+$ ]] || (( TASK_ID < 0 || TASK_ID > 9 )); then
        echo "Invalid D1 task id: ${TASK_ID}" >&2
        exit 2
    fi
done

worker_status=0

for TASK_ID in "${TASK_IDS[@]}"; do
    TASK_TAG="$(printf 'task_%02d' "${TASK_ID}")"
    SPEC="${HERE}/specs/${SUITE}/${TASK_TAG}/${CONDITION}.json"
    TASK_BASE_ROOT="${RUN_ROOT}/${TASK_TAG}"
    if [[ "${TRIAL_START}" -eq 0 && "${TRIAL_COUNT}" -eq 1 ]]; then
        TASK_ROOT="${TASK_BASE_ROOT}"
    else
        TASK_ROOT="${TASK_BASE_ROOT}/trials_$(printf '%02d' "${TRIAL_START}")_$(printf '%02d' "$((TRIAL_START + TRIAL_COUNT - 1))")"
    fi
    RESULT="${TASK_ROOT}/${SUITE}/gpu${GPU_ID}_task${TASK_ID}_results.json"
    mkdir -p "${TASK_ROOT}/logs"
    if [[ ! -f "${SPEC}" ]]; then
        printf '%s\t%s\tmissing_spec\n' "$(date -Is)" "${TASK_TAG}" >> "${STATUS}"
        worker_status=1
        continue
    fi
    export FASTWAM_ENV_EDIT_SPEC="${SPEC}"
    printf '%s\t%s\tstart\n' "$(date -Is)" "${TASK_TAG}" >> "${STATUS}"
    cd "${FASTWAM_ROOT}"
    if timeout "${D1_TASK_TIMEOUT:-60m}" "${FASTWAM_PYTHON}" "$LIBERO_VPRO_FASTWAM_RUNNER" \
        task=libero_uncond_2cam224_1e-4 \
        ckpt="$FASTWAM_CHECKPOINT" \
        EVALUATION.dataset_stats_path="$FASTWAM_DATASET_STATS" \
        EVALUATION.task_suite_name="${SUITE}" \
        EVALUATION.task_id="${TASK_ID}" \
        ++EVALUATION.trial_start="${TRIAL_START}" \
        EVALUATION.num_trials="${TRIAL_COUNT}" \
        EVALUATION.output_dir="${TASK_ROOT}" \
        EVALUATION.device=cuda \
        EVALUATION.num_inference_steps=10 \
        model.skip_dit_load_from_pretrain=true \
        gpu_id="${GPU_ID}" \
        > "${TASK_ROOT}/logs/eval_${RUN_TAG}.log" 2>&1; then
        printf '%s\t%s\tdone\n' "$(date -Is)" "${TASK_TAG}" >> "${STATUS}"
    else
        rc=$?
        printf '%s\t%s\terror:%s\n' "$(date -Is)" "${TASK_TAG}" "${rc}" >> "${STATUS}"
        worker_status=1
    fi
done

exit "${worker_status}"

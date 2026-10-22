#!/usr/bin/env bash

set -Eeuo pipefail
if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 GPU_ID WORKER_RANK NUM_WORKERS" >&2
    exit 2
fi

GPU_ID="$1"
WORKER_RANK="$2"
NUM_WORKERS="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${B1_OUTPUT_ROOT:-${HERE}/../../outputs/B1}"
TRIALS="${LIBERO_VPRO_TRIALS:-${B1_TRIALS:-10}}"
TRIAL_START="${LIBERO_VPRO_TRIAL_START:-0}"
RUN_ROOT="${OUTPUT_ROOT}/fastwam"
LOG_ROOT="${OUTPUT_ROOT}/launcher_logs"
STATUS="${RUN_ROOT}/worker_${WORKER_RANK}.status.tsv"

source "$HERE/../../scripts/fastwam_env.sh"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
export WAM_RECORD_EPISODES=1
export WAM_RECORD_PROFILE=compact
export WAM_RECORD_COMPRESS=1
export WAM_EPISODE_RECORD_ROOT="${OUTPUT_ROOT}/episode_records"
export WAM_TRACK=B1
export WAM_SUBTASK=B1
export WAM_CONDITION=perturbed
unset FASTWAM_OBS_OCCLUSION_SPEC LIBERO_OBS_OCCLUSION_SPEC
unset LIBERO_ENV_SEED

SUITES=(libero_object libero_spatial libero_goal libero_10)
[[ -z "${LIBERO_VPRO_SUITE:-}" ]] || SUITES=("$LIBERO_VPRO_SUITE")
TASKS=($(seq 0 9))
[[ -z "${LIBERO_VPRO_TASK:-}" ]] || TASKS=("$LIBERO_VPRO_TASK")
CONDITIONS=(
    B1_replay_agent B1_replay_wrist
    B1_freeze_agent B1_freeze_wrist
    B1_delay1_agent B1_delay1_wrist
    B1_delay3_agent B1_delay3_wrist
    B1_delay5_agent B1_delay5_wrist
)
if [[ -n "${B1_ONLY_CONDITION:-}" ]]; then
    CONDITIONS=("${B1_ONLY_CONDITION}")
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
touch "${STATUS}"

job_index=0
worker_status=0
for SUITE in "${SUITES[@]}"; do
    for CONDITION in "${CONDITIONS[@]}"; do
        SCHEDULE="${HERE}/configs/trackB_${CONDITION/replay_/replayv2_}.json"
        for TASK_ID in "${TASKS[@]}"; do
            if (( job_index % NUM_WORKERS != WORKER_RANK )); then
                job_index=$((job_index + 1))
                continue
            fi
            job_index=$((job_index + 1))
            TASK_TAG="$(printf 'task_%02d' "${TASK_ID}")"
            TASK_ROOT="${RUN_ROOT}/${SUITE}/${CONDITION}/${TASK_TAG}"
            mkdir -p "${TASK_ROOT}/logs"
            if [[ ! -f "${SCHEDULE}" ]]; then
                printf '%s\t%s\t%s\t%s\tmissing_schedule\n' "$(date -Is)" "${SUITE}" "${CONDITION}" "${TASK_TAG}" >> "${STATUS}"
                worker_status=1
                continue
            fi
            export FASTWAM_DECISION_SCHEDULE_SPEC="${SCHEDULE}"
            export WAM_CASE_ID="${CONDITION}"
            printf '%s\t%s\t%s\t%s\tstart\n' "$(date -Is)" "${SUITE}" "${CONDITION}" "${TASK_TAG}" >> "${STATUS}"
            cd "${FASTWAM_ROOT}"
            if timeout "${B1_TASK_TIMEOUT:-240m}" "${FASTWAM_PYTHON}" "$LIBERO_VPRO_FASTWAM_RUNNER" \
                task=libero_uncond_2cam224_1e-4 \
                ckpt="$FASTWAM_CHECKPOINT" \
                EVALUATION.dataset_stats_path="$FASTWAM_DATASET_STATS" \
                EVALUATION.task_suite_name="${SUITE}" \
                EVALUATION.task_id="${TASK_ID}" \
                ++EVALUATION.trial_start="${TRIAL_START}" \
                EVALUATION.num_trials="${TRIALS}" \
                EVALUATION.output_dir="${TASK_ROOT}" \
                EVALUATION.device=cuda \
                EVALUATION.num_inference_steps=10 \
                model.skip_dit_load_from_pretrain=true \
                gpu_id="${GPU_ID}" \
                > "${TASK_ROOT}/logs/eval.log" 2>&1; then
                printf '%s\t%s\t%s\t%s\tdone\n' "$(date -Is)" "${SUITE}" "${CONDITION}" "${TASK_TAG}" >> "${STATUS}"
            else
                rc=$?
                printf '%s\t%s\t%s\t%s\terror:%s\n' "$(date -Is)" "${SUITE}" "${CONDITION}" "${TASK_TAG}" "${rc}" >> "${STATUS}"
                worker_status=1
            fi
        done
    done
done

exit "${worker_status}"

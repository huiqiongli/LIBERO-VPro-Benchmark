#!/usr/bin/env bash
set -Eeuo pipefail

GPU="${1:?GPU_ID}"
SUITE="${2:?LIBERO_SUITE}"
N="${3:?1|2|3}"
TRIALS="${LIBERO_VPRO_TRIALS:-${4:-10}}"
TRIAL_START="${LIBERO_VPRO_TRIAL_START:-0}"
[[ -z "${LIBERO_VPRO_SUITE:-}" || "$SUITE" == "$LIBERO_VPRO_SUITE" ]] || exit 0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${A4_TARGET_NOISE_OUTPUT_ROOT:?set A4_TARGET_NOISE_OUTPUT_ROOT}"
OUT="$OUT_ROOT/fastwam/${SUITE}/noise_${N}replan"
source "$HERE/../../scripts/fastwam_env.sh"
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WAM_RECORD_EPISODES=1 WAM_RECORD_PROFILE=a4_compact WAM_RECORD_COMPRESS=1
export WAM_TRACK=A4 WAM_SUBTASK=temporary_observation_corruption WAM_CONDITION=perturbed
export WAM_EPISODE_RECORD_ROOT="$OUT/episode_records"
export LIBERO_OBS_OCCLUSION_SPEC="$HERE/../../paper_facing_configs/A4/target_region_noise.json"
export LIBERO_SCHEDULE_SPEC="$HERE/configs/trackA_A4rand_${N}replan.json"
mkdir -p "$OUT"
cd "$FASTWAM_ROOT"

tasks=( $(seq 0 9) )
[[ -z "${LIBERO_VPRO_TASK:-}" ]] || tasks=("$LIBERO_VPRO_TASK")
if [[ "${A4_SMOKE:-0}" == 1 ]]; then
  tasks=("${A4_SMOKE_TASK:-0}")
  TRIALS="${A4_SMOKE_TRIALS:-1}"
fi
for task in "${tasks[@]}"; do
  task_out="$OUT/task_$(printf '%02d' "$task")"
  mkdir -p "$task_out"
  "$FASTWAM_PYTHON" "$LIBERO_VPRO_FASTWAM_RUNNER" \
    task=libero_uncond_2cam224_1e-4 \
    ckpt="$FASTWAM_CHECKPOINT" \
    EVALUATION.dataset_stats_path="$FASTWAM_DATASET_STATS" \
    EVALUATION.task_suite_name="$SUITE" EVALUATION.task_id="$task" \
    ++EVALUATION.trial_start="$TRIAL_START" EVALUATION.num_trials="$TRIALS" \
    EVALUATION.output_dir="$task_out" EVALUATION.device=cuda \
    EVALUATION.num_inference_steps=10 gpu_id="$GPU"
done

#!/usr/bin/env bash
set -Eeuo pipefail

GPU=${1:?gpu id}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${A4_OUTPUT_ROOT:-$HERE/../../outputs/A4}
source "$HERE/../../scripts/fastwam_env.sh"
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU
export WAM_RECORD_EPISODES=1 WAM_RECORD_PROFILE=a4_compact WAM_RECORD_COMPRESS=1
export WAM_EPISODE_RECORD_ROOT=$OUT/episode_records WAM_TRACK=A4 WAM_CONDITION=perturbed
mkdir -p "$OUT"
cd "$FASTWAM_ROOT"

run_condition() {
  local kind="$1" label="$2" schedule="$3"
  local obs
  if [[ "$kind" == noise ]]; then
    obs="$HERE/configs/trackA_A4obs_fullscreen_noise_v2.json"
  else
    obs="$HERE/configs/trackA_A4obs_fullscreen_black.json"
  fi
  local suites=(libero_object libero_spatial libero_goal libero_10)
  local tasks=( $(seq 0 9) )
  local trial_start=${LIBERO_VPRO_TRIAL_START:-0}
  local trial_count=${LIBERO_VPRO_TRIALS:-10}
  local trials=( $(seq "$trial_start" "$((trial_start + trial_count - 1))") )
  [[ -z "${LIBERO_VPRO_SUITE:-}" ]] || suites=("$LIBERO_VPRO_SUITE")
  [[ -z "${LIBERO_VPRO_TASK:-}" ]] || tasks=("$LIBERO_VPRO_TASK")
  if [[ "${A4_SMOKE:-0}" == 1 ]]; then
    suites=("${A4_SMOKE_SUITE:-libero_10}")
    tasks=("${A4_SMOKE_TASK:-0}")
    trials=( $(seq 0 $(( ${A4_SMOKE_TRIALS:-1} - 1 ))) )
  fi
  for suite in "${suites[@]}"; do
    for task in "${tasks[@]}"; do
      for trial in "${trials[@]}"; do
        local caseid="A4_${label}_${suite}_t$(printf '%02d' "$task")_trial$(printf '%02d' "$trial")"
        local out="$OUT/$suite/$label/task_$(printf '%02d' "$task")/trial_$(printf '%02d' "$trial")"
        mkdir -p "$out"
        export LIBERO_OBS_OCCLUSION_SPEC="$obs" LIBERO_SCHEDULE_SPEC="$schedule"
        timeout 2h "$FASTWAM_PYTHON" "$LIBERO_VPRO_FASTWAM_RUNNER" \
          task=libero_uncond_2cam224_1e-4 \
          ckpt="$FASTWAM_CHECKPOINT" \
          EVALUATION.dataset_stats_path="$FASTWAM_DATASET_STATS" \
          EVALUATION.task_suite_name="$suite" EVALUATION.task_id="$task" \
          ++EVALUATION.trial_start="$trial" EVALUATION.num_trials=1 \
          EVALUATION.output_dir="$out" EVALUATION.device=cuda \
          EVALUATION.num_inference_steps=10 gpu_id="$GPU"
      done
    done
  done
}

if [[ "${A4_ONLY_SETTING:-}" =~ ^(black|noise)_D([123])$ ]]; then
  kind="${BASH_REMATCH[1]}"
  duration="${BASH_REMATCH[2]}"
  schedule="${A4_SCHEDULE_SPEC:-$HERE/configs/schedule_D${duration}.json}"
  run_condition "$kind" "${kind}_D${duration}" "$schedule"
else
  for kind in noise black; do
    for n in 1 2 3; do
      if [[ -n "${A4_ONLY_SETTING:-}" && "${A4_ONLY_SETTING}" != "${kind}_N${n}" && "${A4_ONLY_SETTING}" != "${kind}_${n}replan" ]]; then
        continue
      fi
      run_condition "$kind" "${kind}_N${n}" "$HERE/configs/trackA_A4rand_${n}replan.json"
    done
  done
fi

#!/usr/bin/env bash
set -Eeuo pipefail
SUITE="${1:-libero_10}"
[[ -z "${LIBERO_VPRO_SUITE:-}" || "$SUITE" == "$LIBERO_VPRO_SUITE" ]] || exit 0
GPU="${2:-0}"
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SOURCE/../../scripts/fastwam_env.sh"
PY="$FASTWAM_PYTHON"
OUT="${C2_B2_OUTPUT_DIR:-${C2_B2_OUTPUT_ROOT:-$FASTWAM_ROOT/Outputs/C2_B2_recursive_self_prediction_recorded}/fastwam/$SUITE}"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU" PYOPENGL_PLATFORM=egl
export WAM_RECORD_EPISODES=1 WAM_RECORD_PROFILE=a4_compact WAM_RECORD_COMPRESS=1
export WAM_EPISODE_RECORD_ROOT="$OUT/episode_records"
export WAM_TRACK=C2 WAM_SUBTASK=C2 WAM_CONDITION=recursive_self_prediction WAM_CASE_ID=C2_recursive_self_prediction
cd "$FASTWAM_ROOT"
exec "$PY" "$SOURCE/fastwam_recursive_feedback.py" \
  task=libero_uncond_2cam224_1e-4 \
  ckpt="$FASTWAM_CHECKPOINT" \
  EVALUATION.dataset_stats_path="$FASTWAM_DATASET_STATS" \
  EVALUATION.task_suite_name="$SUITE" EVALUATION.task_id="${LIBERO_VPRO_TASK:-${C2_B2_TASK_ID:-0}}" \
  ++EVALUATION.trial_start="${LIBERO_VPRO_TRIAL_START:-${C2_B2_TRIAL_START:-0}}" EVALUATION.num_trials="${LIBERO_VPRO_TRIALS:-${C2_B2_TRIAL_COUNT:-10}}" \
  EVALUATION.output_dir="$OUT" EVALUATION.device=cuda \
  EVALUATION.replan_steps=10 EVALUATION.visualize_future_video=true \
  model.skip_dit_load_from_pretrain=true gpu_id="$GPU"

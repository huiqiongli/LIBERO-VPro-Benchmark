#!/usr/bin/env bash
set -Eeuo pipefail
GPU_ID=${1:?usage: $0 GPU_ID}
A1_RATIO_OUTPUT_ROOT=${A1_RATIO_OUTPUT_ROOT:?A1_RATIO_OUTPUT_ROOT must be set}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT=$A1_RATIO_OUTPUT_ROOT/fastwam
source "$HERE/../../scripts/fastwam_env.sh"
mkdir -p "$MODEL_ROOT/batch" "$MODEL_ROOT/logs" "$MODEL_ROOT/episode_records"
exec 9>"$MODEL_ROOT/worker.lock"
flock -n 9 || exit 75
export CUDA_VISIBLE_DEVICES=$GPU_ID MUJOCO_EGL_DEVICE_ID=$GPU_ID MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WAM_RECORD_EPISODES=1 WAM_RECORD_PROFILE=a4_compact WAM_RECORD_COMPRESS=1
export WAM_EPISODE_RECORD_ROOT=$MODEL_ROOT/episode_records
export WAM_TRACK=A1 WAM_SUBTASK=physical_target_object_ratio WAM_CONDITION=perturbed
unset WAM_SINGLE_VIEW FASTWAM_SINGLE_VIEW LIBERO_SINGLE_VIEW
unset FASTWAM_OBS_OCCLUSION_SPEC LIBERO_OBS_OCCLUSION_SPEC
unset FASTWAM_DECISION_SCHEDULE_SPEC LIBERO_SCHEDULE_SPEC WAM_PERTURBATION_SPEC
unset WAM_RECURSIVE_FEEDBACK FASTWAM_RECURSIVE_FEEDBACK LIBERO_RECURSIVE_FEEDBACK
cd "$FASTWAM_ROOT"
exec timeout 168h "$FASTWAM_PYTHON" "$LIBERO_VPRO_FASTWAM_RUNNER" \
  task=libero_uncond_2cam224_1e-4 \
  ckpt="$FASTWAM_CHECKPOINT" \
  EVALUATION.dataset_stats_path="$FASTWAM_DATASET_STATS" \
  EVALUATION.task_suite_name=libero_object EVALUATION.task_id=0 \
  ++EVALUATION.trial_start=0 EVALUATION.num_trials=1 \
  EVALUATION.output_dir="$MODEL_ROOT/batch" EVALUATION.device=cuda \
  EVALUATION.num_inference_steps=10 \
  +EVALUATION.batch_manifest="${A1_SETTING_MANIFEST:?set A1_SETTING_MANIFEST to the selected trial manifest}" \
  +EVALUATION.batch_shard_id=0 +EVALUATION.batch_num_shards=1 \
  model.skip_dit_load_from_pretrain=true gpu_id="$GPU_ID"

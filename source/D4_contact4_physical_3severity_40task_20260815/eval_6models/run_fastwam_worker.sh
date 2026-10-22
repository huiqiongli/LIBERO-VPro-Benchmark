#!/usr/bin/env bash
set -Eeuo pipefail
GPU_ID=${1:?usage: $0 GPU_ID}
D4_OUTPUT_ROOT=${D4_OUTPUT_ROOT:?D4_OUTPUT_ROOT must be set}
D4_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_ROOT=$D4_OUTPUT_ROOT/fastwam
source "$D4_ROOT/../../scripts/fastwam_env.sh"
export CUDA_VISIBLE_DEVICES=$GPU_ID MUJOCO_EGL_DEVICE_ID=$GPU_ID
export WAM_RECORD_EPISODES=1 WAM_RECORD_PROFILE=a4_compact WAM_RECORD_COMPRESS=1
export WAM_EPISODE_RECORD_ROOT=$MODEL_ROOT/episode_records
export WAM_TRACK=D4 WAM_SUBTASK=D4 WAM_CONDITION=perturbed
unset WAM_SINGLE_VIEW FASTWAM_SINGLE_VIEW LIBERO_SINGLE_VIEW
unset FASTWAM_ENV_EDIT_SPEC LIBERO_ENV_EDIT_SPEC
unset FASTWAM_OBS_OCCLUSION_SPEC LIBERO_OBS_OCCLUSION_SPEC
unset FASTWAM_DECISION_SCHEDULE_SPEC LIBERO_SCHEDULE_SPEC WAM_PERTURBATION_SPEC
unset WAM_RECURSIVE_FEEDBACK FASTWAM_RECURSIVE_FEEDBACK LIBERO_RECURSIVE_FEEDBACK
mkdir -p "$MODEL_ROOT/batch" "$MODEL_ROOT/logs"
cd "$FASTWAM_ROOT"
exec timeout 72h "$FASTWAM_PYTHON" "$LIBERO_VPRO_FASTWAM_RUNNER" \
  task=libero_uncond_2cam224_1e-4 \
  ckpt="$FASTWAM_CHECKPOINT" \
  EVALUATION.dataset_stats_path="$FASTWAM_DATASET_STATS" \
  EVALUATION.task_suite_name=libero_object EVALUATION.task_id=0 \
  ++EVALUATION.trial_start=0 EVALUATION.num_trials=1 \
  EVALUATION.output_dir="$MODEL_ROOT/batch" EVALUATION.device=cuda \
  EVALUATION.num_inference_steps=10 \
  +EVALUATION.batch_manifest="${D4_SETTING_MANIFEST:-$D4_ROOT/fastwam_batch_manifest.json}" \
  +EVALUATION.batch_shard_id=0 +EVALUATION.batch_num_shards=1 \
  model.skip_dit_load_from_pretrain=true gpu_id="$GPU_ID"

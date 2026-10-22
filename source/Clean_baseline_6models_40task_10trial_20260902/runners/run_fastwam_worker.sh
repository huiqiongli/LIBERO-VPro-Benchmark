#!/usr/bin/env bash
set -Eeuo pipefail
GPU_ID=${1:?usage: $0 GPU_ID}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HERE/../../scripts/fastwam_env.sh"
OUT=${CLEAN_OUTPUT_ROOT:-${LIBERO_VPRO_OUTPUT_ROOT:-$LIBERO_VPRO_RELEASE_ROOT/../Outputs/libero_vpro/CLEAN}}/fastwam
mkdir -p "$OUT/episode_records" "$OUT/logs"
MANIFEST=$(python3 - "$HERE/fastwam_manifest.json" "$OUT" <<'PY'
import argparse
from pathlib import Path
import sys
from runnable_settings.runner_common import Selection, selected_manifest
print(selected_manifest(Path(sys.argv[1]), Path(sys.argv[2]), Selection.from_args(argparse.Namespace()), dry_run=False))
PY
)
export CUDA_VISIBLE_DEVICES=$GPU_ID MUJOCO_EGL_DEVICE_ID=$GPU_ID
export WAM_RECORD_EPISODES=1 WAM_RECORD_PROFILE=a4_compact WAM_RECORD_COMPRESS=1
export WAM_EPISODE_RECORD_ROOT=$OUT/episode_records WAM_TRACK=CLEAN WAM_SUBTASK=clean_baseline WAM_CONDITION=clean WAM_CASE_ID=clean_baseline
unset WAM_SINGLE_VIEW FASTWAM_SINGLE_VIEW LIBERO_SINGLE_VIEW WAM_VIEW_COPY
unset WAM_A3_SPEC FASTWAM_ENV_EDIT_SPEC LIBERO_ENV_EDIT_SPEC
unset FASTWAM_OBS_OCCLUSION_SPEC LIBERO_OBS_OCCLUSION_SPEC FASTWAM_DECISION_SCHEDULE_SPEC LIBERO_SCHEDULE_SPEC WAM_PERTURBATION_SPEC
unset WAM_RECURSIVE_FEEDBACK FASTWAM_RECURSIVE_FEEDBACK LIBERO_RECURSIVE_FEEDBACK A4_KIND A4_TARGET_PHASE B1_MODE B1_TARGET_VIEW
cd "$FASTWAM_ROOT"
timeout 168h "$FASTWAM_PYTHON" "$LIBERO_VPRO_FASTWAM_RUNNER" \
  task=libero_uncond_2cam224_1e-4 \
  ckpt="$FASTWAM_CHECKPOINT" \
  EVALUATION.dataset_stats_path="$FASTWAM_DATASET_STATS" \
  EVALUATION.task_suite_name=libero_object EVALUATION.task_id=0 \
  ++EVALUATION.trial_start=0 EVALUATION.num_trials=1 \
  EVALUATION.output_dir="$OUT/batch" EVALUATION.device=cuda \
  EVALUATION.replan_steps=10 EVALUATION.num_steps_wait=30 \
  EVALUATION.num_inference_steps=10 EVALUATION.visualize_future_video=false \
  +EVALUATION.batch_manifest="$MANIFEST" +EVALUATION.batch_shard_id=0 +EVALUATION.batch_num_shards=1 \
  model.skip_dit_load_from_pretrain=true gpu_id="$GPU_ID" \
  > "$OUT/logs/eval.log" 2>&1
date -Is > "$OUT/COMPLETE"

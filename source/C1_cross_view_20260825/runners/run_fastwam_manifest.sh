#!/usr/bin/env bash
set -Eeuo pipefail

MANIFEST=${1:?usage: $0 MANIFEST GPU_ID OUTPUT_ROOT}
GPU_ID=${2:?usage: $0 MANIFEST GPU_ID OUTPUT_ROOT}
OUTPUT_ROOT=${3:?usage: $0 MANIFEST GPU_ID OUTPUT_ROOT}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HERE/../../scripts/fastwam_env.sh"
FASTWAM=$FASTWAM_ROOT
PY=$FASTWAM_PYTHON
CKPT=$FASTWAM_CHECKPOINT
STATS=$FASTWAM_DATASET_STATS

for required in "$MANIFEST" "$PY" "$CKPT" "$STATS"; do
  [[ -e "$required" ]] || { echo "missing FastWAM input: $required" >&2; exit 2; }
done
mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/episode_records"
export CUDA_VISIBLE_DEVICES=$GPU_ID MUJOCO_EGL_DEVICE_ID=$GPU_ID
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1
export WAM_RECORD_EPISODES=1 WAM_RECORD_PROFILE=a4_compact WAM_RECORD_COMPRESS=1
export WAM_EPISODE_RECORD_ROOT=$OUTPUT_ROOT/episode_records
unset WAM_STATE_CASE_JSON
unset WAM_SINGLE_VIEW FASTWAM_SINGLE_VIEW LIBERO_SINGLE_VIEW
unset FASTWAM_OBS_OCCLUSION_SPEC LIBERO_OBS_OCCLUSION_SPEC
unset FASTWAM_DECISION_SCHEDULE_SPEC LIBERO_SCHEDULE_SPEC WAM_PERTURBATION_SPEC
unset WAM_RECURSIVE_FEEDBACK FASTWAM_RECURSIVE_FEEDBACK LIBERO_RECURSIVE_FEEDBACK
unset A4_KIND A4_TARGET_PHASE B1_MODE B1_TARGET_VIEW

cd "$FASTWAM"
"$PY" "$HERE/isolated_sources/fastwam/experiments/libero/eval_libero_single.py" \
  task=libero_uncond_2cam224_1e-4 \
  ckpt="$CKPT" \
  EVALUATION.dataset_stats_path="$STATS" \
  EVALUATION.task_suite_name=libero_object EVALUATION.task_id=0 \
  ++EVALUATION.trial_start=0 EVALUATION.num_trials=1 \
  EVALUATION.output_dir="$OUTPUT_ROOT/batch" EVALUATION.device=cuda \
  EVALUATION.replan_steps=10 EVALUATION.num_steps_wait=30 \
  EVALUATION.num_inference_steps=10 EVALUATION.visualize_future_video=false \
  +EVALUATION.batch_manifest="$MANIFEST" \
  +EVALUATION.batch_shard_id=0 +EVALUATION.batch_num_shards=1 \
  model.skip_dit_load_from_pretrain=true gpu_id="$GPU_ID"

# FastWAM evaluates the whole manifest under one resident model process.  Do
# not accept that batch-level exit status alone: every row must independently
# satisfy the same record/video contract enforced by the other five runners.
"$PY" - "$MANIFEST" "$OUTPUT_ROOT/validation_rows.tsv" <<'PY'
import json
from pathlib import Path
import sys
payload = json.loads(Path(sys.argv[1]).read_text())
rows = payload.get("episodes") or payload.get("cases") or []
if not rows:
    raise SystemExit("manifest has no selected episodes")
text = "".join(f"{row.get('case_json', row.get('spec'))}\t{row['trial_index']}\t{row['case_id']}\n" for row in rows)
Path(sys.argv[2]).write_text(text)
PY
while IFS=$'\t' read -r CASE_JSON TRIAL CASE_ID; do
  TAG="${CASE_ID}__trial$(printf '%02d' "$TRIAL")"
  CASE_ROOT="$OUTPUT_ROOT/batch/cases/$TAG"
  "$PY" "$HERE/validate_episode_output.py" \
    --model fastwam --case-json "$CASE_JSON" --trial "$TRIAL" \
    --record-root "$OUTPUT_ROOT/episode_records" \
    --video-root "$CASE_ROOT/videos" \
    --report "$CASE_ROOT/output_validation.json"
done <"$OUTPUT_ROOT/validation_rows.tsv"

date -Is >"$OUTPUT_ROOT/COMPLETE"

#!/usr/bin/env bash
set -Eeuo pipefail

GPU=${1:?gpu id}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${A4_PHASE_FULL_ROOT:?set A4_PHASE_FULL_ROOT}/fastwam
CFG=$HERE/configs
source "$HERE/../../scripts/fastwam_env.sh"
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WAM_RECORD_EPISODES=1 WAM_RECORD_PROFILE=a4_compact WAM_RECORD_COMPRESS=1
export WAM_EPISODE_RECORD_ROOT=$OUT/episode_records WAM_TRACK=A4
export WAM_SUBTASK=phase_3replan_once WAM_CONDITION=perturbed
case "${A4_PHASE_KIND:-noise}" in
  black) OBS_SPEC=$CFG/trackA_A4obs_fullscreen_black.json ;;
  noise) OBS_SPEC=$CFG/trackA_A4obs_fullscreen_noise_v2.json ;;
  *) echo "unsupported A4 phase corruption kind: $A4_PHASE_KIND (expected black or noise)" >&2; exit 2 ;;
esac
[[ -f "$OBS_SPEC" ]] || { echo "missing observation spec: $OBS_SPEC" >&2; exit 2; }
export LIBERO_OBS_OCCLUSION_SPEC=$OBS_SPEC
mkdir -p "$OUT/logs" "$OUT/episode_records"
cd "$FASTWAM_ROOT"
read -r -a PHASES <<< "${A4_PHASES:-reach pre_grasp grasp transport pre_place}"
for phase in "${PHASES[@]}"; do
  phase_out=$OUT/$phase
  mkdir -p "$phase_out"
  manifest=$(python3 - "$CFG/phase_full/$phase.json" "$phase_out" <<'PY'
import argparse
from pathlib import Path
import sys
from runnable_settings.runner_common import Selection, selected_manifest
selection = Selection.from_args(argparse.Namespace())
print(selected_manifest(Path(sys.argv[1]), Path(sys.argv[2]), selection, dry_run=False))
PY
  )
  export WAM_CASE_ID=A4_phase_${phase}_3replan_once_full
  export LIBERO_SCHEDULE_SPEC=$CFG/phase_once/$phase.json
  "$FASTWAM_PYTHON" "$LIBERO_VPRO_FASTWAM_RUNNER" \
    task=libero_uncond_2cam224_1e-4 \
    ckpt="$FASTWAM_CHECKPOINT" \
    EVALUATION.dataset_stats_path="$FASTWAM_DATASET_STATS" \
    EVALUATION.task_suite_name=libero_object EVALUATION.task_id=0 EVALUATION.num_trials=1 \
    +EVALUATION.batch_manifest="$manifest" \
    +EVALUATION.batch_shard_id=0 +EVALUATION.batch_num_shards=1 \
    EVALUATION.output_dir="$phase_out" EVALUATION.device=cuda \
    EVALUATION.num_inference_steps=10 model.skip_dit_load_from_pretrain=true gpu_id=$GPU \
    >"$OUT/logs/${phase}.log" 2>&1
done
date -Is > "$OUT/COMPLETE"

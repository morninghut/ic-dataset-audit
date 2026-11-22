#!/bin/bash
# The unattended GPU queue of step 3, in the order the single RTX 4090 allows.
#
#   pipeline.sh [seed]
#
#   1. wait for the three DENEB trainings that are already running
#   2. score every DENEB variant on the correlation and PASCAL splits (three at once, ~5 GB each)
#   3. check that Pearl's image fast path changes nothing, then train and score the
#      three Pearl variants one after the other (one run needs ~14.5 GB)
#   4. the vision-language judge, without and with references, on the same splits
#   5. FOIL (398k rows) for DENEB and the judge, last because nothing else depends on it
#
# Every stage logs to runs/metrics/logs/ and a stage that fails does not stop
# the later ones; pipeline.log records what ran and how it ended.
set -u

SEED=${1:-12}
ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
S3=$ROOT/analysis/metrics
RUNS=$ROOT/runs/metrics
LOG=$RUNS/logs
VARIANTS="released pairclean randomdrop"
CORE="spica_test polaris_test nebula_test flickr8k_expert composite flickr8k_cf pascal50s polaris_val nebula_valid spica_val"
FOIL="foil_1ref foil_4ref"

say() { echo "[$(date '+%F %T')] $*" >> "$LOG/pipeline.log"; }
stage() {  # stage <name> <command...>: run, log the exit status
  local name=$1; shift
  say "start $name"
  "$@" > "$LOG/$name.log" 2>&1
  say "end   $name (exit $?)"
}

score_deneb() {  # score_deneb <variant> <splits...>
  local variant=$1; shift
  local run=$RUNS/deneb/${variant}_s$SEED
  local checkpoint
  checkpoint=$("$ROOT/external/envs/deneb/bin/python" "$S3/best_checkpoint.py" "$run" 2>> "$LOG/pipeline.log")
  say "deneb $variant scores $checkpoint"
  (cd "$run" && PYTHONPATH=$ROOT/external/DENEB "$ROOT/external/envs/deneb/bin/python" \
      "$S3/score_deneb.py" "$checkpoint" "$run/scores" "$@")
}

score_pearl() {  # score_pearl <variant>: both text lookups, every split
  local run=$RUNS/pearl/${1}_s$SEED
  local checkpoint
  checkpoint=$("$ROOT/external/envs/pearl/bin/python" "$S3/best_checkpoint.py" "$run" 2>> "$LOG/pipeline.log")
  say "pearl $1 scores $checkpoint"
  (cd "$run" && PYTHONPATH=$ROOT/external/Pearl "$ROOT/external/envs/pearl/bin/python" \
      "$S3/score_pearl.py" "$checkpoint" "$run/scores_raw" $CORE $FOIL &&
   PYTHONPATH=$ROOT/external/Pearl "$ROOT/external/envs/pearl/bin/python" \
      "$S3/score_pearl.py" "$checkpoint" "$run/scores_prefixed" --prefixed $CORE $FOIL)
}

vlm() {  # vlm <mode> <splits...>
  local mode=$1; shift
  HF_HUB_OFFLINE=1 "$ROOT/external/envs/vlm/bin/python" "$S3/score_vlm.py" "$mode" 32 "$@"
}

say "pipeline started, seed $SEED"
while pgrep -f "released_cli.py deneb train" > /dev/null; do sleep 30; done
say "deneb trainings finished"

for V in $VARIANTS; do stage "score_deneb_${V}_s$SEED" score_deneb "$V" $CORE & done
wait

stage fastpath_check bash "$S3/check_pearl_fastpath.sh"
for V in $VARIANTS; do
  stage "pearl_${V}_s$SEED" bash "$S3/run_pearl.sh" "$V" "$SEED"
  stage "score_pearl_${V}_s$SEED" score_pearl "$V"
done

stage vlm_free vlm free $CORE
stage vlm_refs vlm refs $CORE

for V in $VARIANTS; do stage "score_deneb_foil_${V}_s$SEED" score_deneb "$V" $FOIL & done
wait
stage vlm_free_foil vlm free $FOIL
stage vlm_refs_foil vlm refs $FOIL
say "pipeline finished"

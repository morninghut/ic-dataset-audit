#!/bin/bash
# The unattended queue of step 3 on the single RTX 4090.
#
#   nohup bash analysis/metrics/queue.sh [seed=12] [extra seed=13] > /dev/null 2>&1 &
#
# Phase A, what the report needs (ends with runs/metrics/PHASE_A_DONE):
#   1. wait for the DENEB trainings of <seed> that are already running
#   2. score the three DENEB variants on the correlation and PASCAL splits (three at once, ~5 GB each)
#   3. check Pearl's image fast path, then train the three Pearl variants one after the
#      other (a run wants the card to itself) and score each with both text lookups
#   4. the vision-language judge without and with references on the same splits
#   5. register everything and run the checks, the leakage contrasts and protocol.py
# Phase B, extras that only use idle time and may be interrupted (ends with PHASE_B_DONE):
#   6. FOIL for the three DENEB variants
#   7. the same DENEB and Pearl trainings with <extra seed>, scored like 2 and 3
#   8. the judge without references on FOIL
#   9. step 5 again, without protocol.py (it uses neither FOIL nor the extra seed)
# The classic metrics (BLEU, ROUGE-L, CIDEr, METEOR, SPICE) run on 32 CPU cores at low
# priority next to all of this, first the correlation splits and then FOIL.
#
# Every stage appends to runs/metrics/logs/<stage>.log, is retried once unless it
# timed out, and never stops the stages after it.  logs/pipeline.log records what ran,
# when, and how it ended.  Scoring scripts skip splits they already wrote, so starting
# the queue again after an interruption repeats only the trainings.
set -u

SEED=${1:-12}
EXTRA=${2:-13}
ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
S3=$ROOT/analysis/metrics
RUNS=$ROOT/runs/metrics
LOG=$RUNS/logs
VLM_OUT=$RUNS/vlm/Qwen3-VL-4B-Instruct
VARIANTS="released pairclean randomdrop"
CORE="spica_test polaris_test nebula_test flickr8k_expert composite flickr8k_cf pascal50s polaris_val nebula_valid spica_val"
FOIL="foil_1ref foil_4ref"
mkdir -p "$LOG"

say() { echo "[$(date '+%F %T')] $*" >> "$LOG/pipeline.log"; }

stage() {  # stage <name> <command...>
  local name=$1 code=1 attempt; shift
  for attempt in 1 2; do
    say "start $name (attempt $attempt)"
    echo "===== $(date '+%F %T') attempt $attempt: $*" >> "$LOG/$name.log"
    "$@" >> "$LOG/$name.log" 2>&1
    code=$?
    say "end   $name (exit $code)"
    [ $code -eq 0 ] && return 0
    [ $code -eq 124 ] && return 124   # timed out: a second attempt would only take as long again
    sleep 60
  done
  return $code
}

wait_gpu() {  # wait_gpu <MiB>: wait up to 30 minutes for that much free memory, then go on anyway
  local free=0 i
  for i in $(seq 60); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "${free:-0}" -ge "$1" ] && return 0
    sleep 30
  done
  say "gpu: $free MiB free after 30 minutes, wanted $1; going on"
}

train() {  # train <deneb|pearl> <variant> <seed>
  timeout -k 60 16h bash "$S3/run_$1.sh" "$2" "$3"
}

best() {  # best <deneb|pearl> <run dir>: absolute path of the checkpoint the trainer ranked first
  local checkpoint
  checkpoint=$(cd "$2" && "$ROOT/external/envs/$1/bin/python" "$S3/best_checkpoint.py" "$2") || return 1
  checkpoint=$(cd "$2" && realpath "$checkpoint")
  [ -f "$checkpoint" ] || { echo "no checkpoint in $2: $checkpoint"; return 1; }
  echo "$checkpoint"
}

score_deneb() {  # score_deneb <variant> <seed> <splits...>
  local run=$RUNS/deneb/${1}_s$2 checkpoint; shift 2
  checkpoint=$(best deneb "$run") || { echo "$checkpoint"; return 1; }
  echo "checkpoint $checkpoint"
  (cd "$run" && PYTHONPATH=$ROOT/external/DENEB timeout -k 60 6h "$ROOT/external/envs/deneb/bin/python" \
      "$S3/score_deneb.py" "$checkpoint" "$run/scores" "$@")
}

score_pearl() {  # score_pearl <variant> <seed>: the training-side and the evaluation-side text lookup
  local run=$RUNS/pearl/${1}_s$2 checkpoint code=0
  checkpoint=$(best pearl "$run") || { echo "$checkpoint"; return 1; }
  echo "checkpoint $checkpoint"
  (cd "$run" && PYTHONPATH=$ROOT/external/Pearl timeout -k 60 4h "$ROOT/external/envs/pearl/bin/python" \
      "$S3/score_pearl.py" "$checkpoint" "$run/scores_raw" $CORE $FOIL) || code=$?
  (cd "$run" && PYTHONPATH=$ROOT/external/Pearl timeout -k 60 4h "$ROOT/external/envs/pearl/bin/python" \
      "$S3/score_pearl.py" "$checkpoint" "$run/scores_prefixed" --prefixed $CORE $FOIL) || code=$?
  return $code
}

vlm() {  # vlm <mode> <splits...>: finished triples are kept, so a smaller batch resumes
  local mode=$1 batch; shift
  for batch in 32 16 8; do
    echo "== batch size $batch"
    HF_HUB_OFFLINE=1 timeout -k 60 12h "$ROOT/external/envs/vlm/bin/python" "$S3/score_vlm.py" "$mode" "$batch" "$@" && return 0
  done
  return 1
}

classics() {  # classics <splits...>: the three groups, on cores 64-95 at the lowest priority
  local group code=0
  for group in ngram meteor spice; do
    (cd "$S3" && taskset -c 64-95 nice -n 19 "$ROOT/external/envs/deneb/bin/python" score_classic.py "$group" "$@") || code=$?
  done
  return $code
}

classics_all() {  # one SPICE at a time: it keeps its parses in one cache directory
  while pgrep -f "score_classic.py" > /dev/null; do sleep 60; done
  stage classic_core classics $CORE
  touch "$RUNS/classic/CORE_ENDED"
  stage classic_foil classics $FOIL
}

columns() {  # columns <directory with %s for the label> <labels...> -> label=directory for those with scores
  local pattern=$1 label directory; shift
  for label in "$@"; do
    directory=$(printf "$pattern" "$label")
    [ -f "$directory/flickr8k_expert.npz" ] && printf '%s=%s ' "$label" "$directory"
  done
}

analysis() (  # analysis <protocol: 0|1> <seeds...>
  protocol=$1; shift
  cd "$S3" || exit 1
  export PYTHONPATH=$ROOT/Polos-revised HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  py="nice -n 19 $ROOT/.venv/bin/python"
  register() {  # register <name> <columns...>: nothing to do without columns
    [ $# -gt 1 ] || { say "register $1: no scores"; return 0; }
    stage "register_$1" $py register.py "$@"
  }
  classic=$RUNS/classic
  register classic "bleu1=$classic/ngram:bleu1" "bleu4=$classic/ngram:bleu4" "rougel=$classic/ngram:rougel" \
      "cider=$classic/ngram:cider" "meteor=$classic/meteor:meteor" "spice=$classic/spice:spice"
  stage check_classic $py check_classic.py
  for seed in "$@"; do
    register "deneb_s$seed" $(columns "$RUNS/deneb/%s_s$seed/scores" $VARIANTS)
    register "pearl_s$seed" $(columns "$RUNS/pearl/%s_s$seed/scores_raw" $VARIANTS)
    register "pearl_s${seed}_prefixed" $(columns "$RUNS/pearl/%s_s$seed/scores_prefixed" $VARIANTS)
    stage "check_deneb_s$seed" $py check_trained.py deneb "deneb_s$seed"
    stage "check_pearl_s$seed" $py check_trained.py pearl "pearl_s$seed"
    stage "check_pearl_s${seed}_prefixed" $py check_trained.py pearl "pearl_s${seed}_prefixed"
    stage "leakage_deneb_s$seed" $py leakage.py "external/deneb_s$seed" nebula_train 1000
    stage "leakage_pearl_s$seed" $py leakage.py "external/pearl_s$seed" spica_train 1000
    stage "leakage_pearl_s${seed}_prefixed" $py leakage.py "external/pearl_s${seed}_prefixed" spica_train 1000
  done
  register vlm $(columns "$VLM_OUT/%s" free refs)
  [ "$protocol" = 1 ] && stage protocol $py protocol.py 1000 20
  exit 0
)

say "queue started, seed $SEED, extra seed $EXTRA (pid $$)"

# ---------------------------------------------------------------- phase A
while pgrep -f "released_cli.py deneb train" > /dev/null; do sleep 30; done
say "deneb trainings of seed $SEED finished"
classics_all &
CLASSICS=$!

wait_gpu 15500
for V in $VARIANTS; do stage "score_deneb_${V}_s$SEED" score_deneb "$V" "$SEED" $CORE & SCORERS="${SCORERS:-} $!"; done
wait $SCORERS

wait_gpu 20000
stage fastpath_check timeout -k 60 4h bash "$S3/check_pearl_fastpath.sh"
for V in $VARIANTS; do
  wait_gpu 20000
  stage "pearl_${V}_s$SEED" train pearl "$V" "$SEED"
  stage "score_pearl_${V}_s$SEED" score_pearl "$V" "$SEED"
done

wait_gpu 20000
stage vlm_free vlm free $CORE
stage vlm_refs vlm refs $CORE

for I in $(seq 180); do [ -f "$RUNS/classic/CORE_ENDED" ] && break; sleep 60; done
analysis 1 "$SEED"
touch "$RUNS/PHASE_A_DONE"
say "phase A finished"

# ---------------------------------------------------------------- phase B
SCORERS=
for V in $VARIANTS; do stage "score_deneb_foil_${V}_s$SEED" score_deneb "$V" "$SEED" $FOIL & SCORERS="$SCORERS $!"; done
wait $SCORERS

wait_gpu 20000
TRAINERS=
for V in $VARIANTS; do stage "deneb_${V}_s$EXTRA" train deneb "$V" "$EXTRA" & TRAINERS="$TRAINERS $!"; sleep 120; done
wait $TRAINERS
SCORERS=
for V in $VARIANTS; do stage "score_deneb_${V}_s$EXTRA" score_deneb "$V" "$EXTRA" $CORE & SCORERS="$SCORERS $!"; done
wait $SCORERS

for V in $VARIANTS; do
  wait_gpu 20000
  stage "pearl_${V}_s$EXTRA" train pearl "$V" "$EXTRA"
  stage "score_pearl_${V}_s$EXTRA" score_pearl "$V" "$EXTRA"
done

wait_gpu 20000
stage vlm_free_foil vlm free $FOIL

wait $CLASSICS
analysis 0 "$SEED" "$EXTRA"
touch "$RUNS/PHASE_B_DONE"
say "phase B finished"

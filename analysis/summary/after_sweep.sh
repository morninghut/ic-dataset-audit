#!/bin/bash
# Wait for deneb_epoch_sweep.sh to finish, then write runs/summary/deneb_epoch_sweep.json.
#   nohup setsid bash analysis/summary/after_sweep.sh > runs/summary/logs/after_sweep.log 2>&1 &
ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
while [ ! -f $ROOT/runs/summary/DENEB_SWEEP_DONE ]; do sleep 60; done
cd $ROOT/analysis/summary
PYTHONPATH=$ROOT/Polos-revised HF_HUB_OFFLINE=1 PYTHONUTF8=1 $ROOT/.venv/bin/python deneb_epoch_report.py
date > $ROOT/runs/summary/DENEB_REPORT_DONE

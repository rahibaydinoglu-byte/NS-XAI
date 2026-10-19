#!/usr/bin/env bash
# Full reproduction of the case study (sequential runs to avoid CPU contention in the cost measurements).
set -e
cd "$(dirname "$0")/code"
for m in ridge tsk tsk8 lgbm mlp transformer lstm; do python run_models.py $m 5 > ../results/log_$m.txt 2>&1; done
python run_xai.py > ../results/log_xai.txt 2>&1
for m in ridge tsk tsk8 lgbm mlp transformer lstm; do TASK=B python run_models.py $m 5 > ../results_B/log_$m.txt 2>&1; done
TASK=B python run_xai.py > ../results_B/log_xai.txt 2>&1
python analysis.py; TASK=B python analysis.py
python fig_framework.py; python figures.py

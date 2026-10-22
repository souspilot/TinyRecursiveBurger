#!/usr/bin/env bash
# End-to-end driver for the 2xA100 server. Run stages individually:
# First: conda env create -f pde_burgers/environment.yml && conda activate trm-burgers
#   bash pde_burgers/run_server.sh setup | data | sanity | bench | train | train_seed1 | eval
# Set EPOCHS_* from the `bench` stage output before `train`.
# Every stage except `setup` runs detached (nohup) and writes only to logs/<name>.log,
# so it survives a closed terminal; follow progress with `tail -f logs/<name>.log`.
set -euo pipefail
cd "$(dirname "$0")/.."

H5=data/pdebench/1D/Burgers/Train/1D_Burgers_Sols_Nu0.01.hdf5
CACHE=data/burgers_nu0.01_t1.npz
EPOCHS_RECURSIVE=${EPOCHS_RECURSIVE:-1000}
EPOCHS_SHALLOW=${EPOCHS_SHALLOW:-1000}
EPOCHS_DEEP=${EPOCHS_DEEP:-1000}
MAX_MIN=${MAX_MIN:-210}
export PYTHONUNBUFFERED=1  # logs update live instead of in buffered chunks
mkdir -p logs

# detach NAME GPU CMD...: run CMD in the background on GPU (or "" for none), output to logs/NAME.log
detach() {
  local name=$1 gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup "$@" > "logs/$name.log" 2>&1 &
  echo "started $name (pid $!) -> tail -f logs/$name.log"
}

case "${1:-}" in
  setup)
    # only needed outside the conda env (environment.yml already installs these)
    pip install -r pde_burgers/requirements.txt ;;
  data)
    detach prepare_data "" bash -c "
      python -m pde_burgers.download --root_folder data/pdebench &&
      python -m pde_burgers.prepare_data --h5 $H5 --out $CACHE --target_time 1.0" ;;
  sanity)
    # tiny-subset overfitting: val relL2 on the same 64 trajectories should fall far below persistence
    for spec in "recursive 0" "shallow 1"; do
      set -- $spec
      detach overfit_$1 $2 python -m pde_burgers.train --model $1 --overfit_n 64 --epochs 5000 \
        --eval_every 100 --warmup_steps 50 --ema_decay 0.99
    done ;;
  bench)
    detach bench 0 bash -c '
      for m in recursive shallow deep; do
        python -m pde_burgers.train --model $m --bench --budget_minutes '"$MAX_MIN"' --out_dir runs/bench_$m
      done' ;;
  train)
    # GPU0: recursive (priority). GPU1: shallow, then the deep control.
    detach recursive_s0 0 python -m pde_burgers.train --model recursive --seed 0 \
      --epochs "$EPOCHS_RECURSIVE" --max_minutes "$MAX_MIN"
    CUDA_VISIBLE_DEVICES=1 nohup bash -c "
      python -m pde_burgers.train --model shallow --seed 0 --epochs $EPOCHS_SHALLOW --max_minutes $MAX_MIN > logs/shallow_s0.log 2>&1
      python -m pde_burgers.train --model deep    --seed 0 --epochs $EPOCHS_DEEP    --max_minutes $MAX_MIN > logs/deep_s0.log 2>&1
    " > /dev/null 2>&1 &
    echo "started shallow_s0 then deep_s0 on GPU1 -> tail -f logs/shallow_s0.log logs/deep_s0.log" ;;
  train_seed1)
    detach recursive_s1 0 python -m pde_burgers.train --model recursive --seed 1 \
      --epochs "$EPOCHS_RECURSIVE" --max_minutes "$MAX_MIN"
    detach shallow_s1 1 python -m pde_burgers.train --model shallow --seed 1 \
      --epochs "$EPOCHS_SHALLOW" --max_minutes "$MAX_MIN" ;;
  eval)
    # run alone on one GPU so latency numbers are not perturbed by other jobs
    detach eval 0 python -m pde_burgers.evaluate --runs 'runs/*_s*' --out results ;;
  *)
    echo "usage: $0 {setup|data|sanity|bench|train|train_seed1|eval}"; exit 1 ;;
esac

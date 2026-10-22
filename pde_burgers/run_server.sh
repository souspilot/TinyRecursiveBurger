#!/usr/bin/env bash
# End-to-end driver for the 2xA100 server. Run stages individually:
# First: conda env create -f pde_burgers/environment.yml && conda activate trm-burgers
#   bash pde_burgers/run_server.sh setup | data | sanity | bench | train | train_seed1 | eval
# Set EPOCHS_* from the `bench` stage output before `train`.
set -euo pipefail
cd "$(dirname "$0")/.."

H5=data/pdebench/1D/Burgers/Train/1D_Burgers_Sols_Nu0.01.hdf5
CACHE=data/burgers_nu0.01_t1.npz
EPOCHS_RECURSIVE=${EPOCHS_RECURSIVE:-1000}
EPOCHS_SHALLOW=${EPOCHS_SHALLOW:-1000}
EPOCHS_DEEP=${EPOCHS_DEEP:-1000}
MAX_MIN=${MAX_MIN:-210}
mkdir -p logs

case "${1:-}" in
  setup)
    # only needed outside the conda env (environment.yml already installs these)
    pip install -r pde_burgers/requirements.txt ;;
  data)
    python -m pde_burgers.download --root_folder data/pdebench
    python -m pde_burgers.prepare_data --h5 "$H5" --out "$CACHE" --target_time 1.0 | tee logs/prepare_data.log ;;
  sanity)
    # tiny-subset overfitting: train relL2 on 64 trajectories should fall far below the persistence error
    CUDA_VISIBLE_DEVICES=0 python -m pde_burgers.train --model recursive --overfit_n 64 --epochs 5000 \
      --eval_every 100 --warmup_steps 50 --ema_decay 0.99 | tee logs/overfit_recursive.log &
    CUDA_VISIBLE_DEVICES=1 python -m pde_burgers.train --model shallow --overfit_n 64 --epochs 5000 \
      --eval_every 100 --warmup_steps 50 --ema_decay 0.99 | tee logs/overfit_shallow.log &
    wait ;;
  bench)
    for m in recursive shallow deep; do
      CUDA_VISIBLE_DEVICES=0 python -m pde_burgers.train --model $m --bench --budget_minutes "$MAX_MIN" \
        --out_dir runs/bench_$m | tee logs/bench_$m.log
    done ;;
  train)
    # GPU0: recursive (priority). GPU1: shallow, then the deep control.
    CUDA_VISIBLE_DEVICES=0 nohup python -m pde_burgers.train --model recursive --seed 0 \
      --epochs "$EPOCHS_RECURSIVE" --max_minutes "$MAX_MIN" > logs/recursive_s0.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 nohup bash -c "
      python -m pde_burgers.train --model shallow --seed 0 --epochs $EPOCHS_SHALLOW --max_minutes $MAX_MIN > logs/shallow_s0.log 2>&1
      python -m pde_burgers.train --model deep    --seed 0 --epochs $EPOCHS_DEEP    --max_minutes $MAX_MIN > logs/deep_s0.log 2>&1
    " > /dev/null 2>&1 &
    echo "launched; follow with: tail -f logs/recursive_s0.log logs/shallow_s0.log logs/deep_s0.log" ;;
  train_seed1)
    CUDA_VISIBLE_DEVICES=0 nohup python -m pde_burgers.train --model recursive --seed 1 \
      --epochs "$EPOCHS_RECURSIVE" --max_minutes "$MAX_MIN" > logs/recursive_s1.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 nohup python -m pde_burgers.train --model shallow --seed 1 \
      --epochs "$EPOCHS_SHALLOW" --max_minutes "$MAX_MIN" > logs/shallow_s1.log 2>&1 &
    echo "launched seed 1" ;;
  eval)
    # run alone on one GPU so latency numbers are not perturbed by other jobs
    CUDA_VISIBLE_DEVICES=0 python -m pde_burgers.evaluate --runs 'runs/*_s*' --out results | tee logs/eval.log ;;
  *)
    echo "usage: $0 {setup|data|sanity|bench|train|train_seed1|eval}"; exit 1 ;;
esac

#!/usr/bin/env bash
# Sequential GPU stages; each service survives the interactive terminal.
set -u
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
unset DBUS_SESSION_BUS_ADDRESS
cd /home/bo/bo01/llmpy311/s2c
while systemctl --user is-active --quiet s2c-banking-fixed-select; do
    sleep 5
done
if ! test -f ../artifacts/s2c/analysis/banking_fixed_known/selection_complete.json; then
    echo 'Known-only selection incomplete; inspect s2c-banking-fixed-select.service'
    exit 1
fi

stage() {
    local unit="$1"
    shift
    systemd-run --user --unit="$unit" --property=RemainAfterExit=yes \
        --property=WorkingDirectory=/home/bo/bo01/llmpy311/s2c \
        --property=MemoryMax=10G --property=MemorySwapMax=512M \
        --property=OOMPolicy=stop --property=CPUQuota=400% --property=Nice=10 \
        --property=StandardOutput=append:/home/bo/bo01/llmpy311/artifacts/s2c/analysis/recovery_stages.log \
        --property=StandardError=inherit \
        --setenv=CONDA_DEFAULT_ENV=bo --setenv=OMP_NUM_THREADS=2 \
        --setenv=OPENBLAS_NUM_THREADS=2 --setenv=MKL_NUM_THREADS=2 \
        --setenv=PYTHONUNBUFFERED=1 "$@" || return 1
    while test "$(systemctl --user show "$unit" -p SubState --value)" = running; do
        sleep 5
    done
    local result
    result="$(systemctl --user show "$unit" -p Result --value)"
    echo "$unit result=$result"
    test "$result" = success
}

stage s2c-banking-fixed-eval /home/bo/anaconda3/envs/bo/bin/python \
    scripts/experiments/evaluate_kir_ours.py --datasets banking77 --kirs .25 .5 .75 \
    --selection-root ../artifacts/s2c/analysis/banking_fixed_known || exit 1
/home/bo/anaconda3/envs/bo/bin/python tools/analysis/build_final_paper_main.py \
    --selection-root ../artifacts/s2c/analysis/banking_fixed_known \
    --output-root results/final_paper_main/fixed_known_control || exit 1

# Verify one previously interrupted baseline before expanding its matrix.
stage s2c-textoir-adb-pilot /home/bo/anaconda3/envs/bo/bin/python \
    scripts/experiments/train_kir_textoir_baselines.py --datasets banking77 \
    --methods ADB --kirs .25 --seeds 13
stage s2c-mogb-recovery /home/bo/anaconda3/envs/bo/bin/python \
    scripts/experiments/run_mogb_shared_official.py --matrix --offload cpu-feature \
    --recompute-feature-graphs
mogb_exit=$?
echo "MOGB service exit=$mogb_exit; resource failure requires inspection before retry"
/home/bo/anaconda3/envs/bo/bin/python tools/analysis/build_final_paper_main.py
exit "$mogb_exit"

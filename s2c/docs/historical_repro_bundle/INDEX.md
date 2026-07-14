# Historical Reproduction Bundle

See also: [TOTAL_LEDGER.md](TOTAL_LEDGER.md)

## Strict Historical Protocol

- `data_root`: `data/v19`
- `known_intents_path`: `data/v19/KNOWN_INTENTS.json`
- `gate_encoder_path`: `all-MiniLM-L6-v2`
- `reference_eval_results`: `outputs/experiments/archive/sweeps/2026-03-23/pipeline_phase3_proto_eval/pipeline_v19_phase3_proto_d_narrow_t085_eval/eval_results.json`

## Replay Status

- `truth_freeze`: `completed`
- `data_validation`: `completed`
- `frozen_eval_replay`: `completed`
- `metric_comparison`: `completed`

## File Status

- `VERIFIED_MAINCHAIN`: `tools/analysis/run_prototype_gate_frozen_baseline_v19.py`
- `VERIFIED_MAINCHAIN`: `tools/eval/eval_system_pipeline_v19.py`
- `VERIFIED_MAINCHAIN`: `tools/analysis/historical_best_pipeline_v19.py`
- `VERIFIED_MAINCHAIN`: `tools/analysis/replay_historical_chain_v19.py`
- `FROZEN_DEPENDENCY`: `outputs/experiments/archive/sweeps/2026-03-23/pipeline_phase3_proto_eval/pipeline_v19_phase3_proto_d_narrow_t085_eval/eval_results.json`
- `FROZEN_DEPENDENCY`: `outputs/experiments/archive/sweeps/2026-03-23/gate_l2_mix2_train/gate_l2_mix2_true_lambda_1p6/detector.json`

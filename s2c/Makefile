.PHONY: install data-import data-build data-validate test audit plan-experiments smoke-experiments gate-grid summarize-results

install:
	python -m pip install -e .

data-import:
	python -m s2c.data.import_textoir

data-build:
	python -m s2c.data.build_canonical
	python -m s2c.data.build_registries
	python -m s2c.data.build_views
	python -m s2c.data.export_protocol

data-validate:
	python -m s2c.data.validate_protocol

test:
	pytest -q

audit:
	python tools/maintenance/check_data_tracking.py
	python tools/maintenance/check_development_log.py

plan-experiments:
	python -m s2c.experiments.plan

smoke-experiments:
	python scripts/experiments/run_gate_grid.py --config configs/experiments/protocol_v2/smoke_gate.yaml

gate-grid:
	python scripts/experiments/run_gate_grid.py --config configs/experiments/protocol_v2/gate_core_dense.yaml

summarize-results:
	python scripts/experiments/summarize_grid.py --config configs/experiments/protocol_v2/gate_core_dense.yaml

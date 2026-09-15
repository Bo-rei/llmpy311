import subprocess
import sys

import pytest

from scripts.experiments import train_kir_textoir_baselines as runner


def arguments(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['runner', '--datasets', 'banking77',
        '--methods', 'ADB', '--kirs', '.25', '--seeds', '13'])


def test_cuda_preflight_failure_prevents_all_attempts(monkeypatch):
    arguments(monkeypatch)
    calls = []
    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(runner.subprocess, 'run', unavailable)
    monkeypatch.setattr(runner, 'run_unit', lambda *args: calls.append(args))
    with pytest.raises(subprocess.CalledProcessError):
        runner.main()
    assert calls == []


def test_failed_cell_produces_failed_matrix_exit(monkeypatch):
    arguments(monkeypatch)
    calls = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: None)
    def fail(*args):
        calls.append(args)
        return 'FAIL banking77/kir25_seed13/ADB exit=1'
    monkeypatch.setattr(runner, 'run_unit', fail)
    assert runner.main() == 1
    assert len(calls) == 1
    assert calls[0][:4] == ('banking77', .25, 13, 'ADB')

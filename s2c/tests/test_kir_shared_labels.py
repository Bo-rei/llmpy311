from pathlib import Path
from types import SimpleNamespace

from tools.compat.textoir.run_external_textoir import install_known_labels


def test_geometry_entrypoint_uses_coverage_selection(monkeypatch):
    from scripts.experiments import run_kir_sensitivity_known_only as runner
    calls = []
    monkeypatch.setattr(runner, 'coverage_geometry', lambda datasets, kirs: calls.append((datasets, kirs)))
    monkeypatch.setattr('sys.argv', ['runner', '--stage', 'geometry', '--datasets', 'banking77', '--kirs', '.25'])
    runner.main()
    assert calls == [(['banking77'], [.25])]


def test_runtime_uses_requested_labels_not_sampled_labels(tmp_path):
    root = tmp_path / 'dataloaders'
    root.mkdir()
    source = root / 'base.py'
    source.write_text('class DataManager:\n'
        '    def __init__(self):\n'
        '        self.known_label_list = ["wrong"]\n'
        '        self.known_label_list = list(self.known_label_list)\n'
        '        self.num_labels = len(self.known_label_list)\n')
    install_known_labels(tmp_path, ['intent_b', 'intent_a'])
    namespace = {}
    exec(compile(source.read_text(), str(source), 'exec'), namespace)
    manager = namespace['DataManager']()
    assert manager.known_label_list == ['intent_b', 'intent_a']
    assert manager.n_known_cls == manager.num_labels == 2

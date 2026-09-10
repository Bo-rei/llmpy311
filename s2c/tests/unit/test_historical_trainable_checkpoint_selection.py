import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel

from scripts.experiments import run_historical_trainable_checkpoint_selection as runner
from scripts.experiments.run_historical_trainable_checkpoint_selection import optimizer_for, set_phase
from protocol_v2.experiments.racal_v1 import representation
from protocol_v2.experiments.racal_v1.representation import RacalMiniLM


def test_projection_only_never_unfreezes_backbone_after_warmup():
    model = RacalMiniLM.__new__(RacalMiniLM)
    torch.nn.Module.__init__(model)
    model.mode = "trainable_projection_only"
    model.encoder = torch.nn.Linear(3, 3)
    model.projection = torch.nn.Linear(3, 3)
    for warmup in (True, False):
        set_phase(model, warmup)
        assert not any(p.requires_grad for p in model.encoder.parameters())
        assert all(p.requires_grad for p in model.projection.parameters())
        optimizer = optimizer_for(model, 0.)
        assert len(optimizer.param_groups) == 1


def test_lora_finetune_parameters_receive_gradients():
    model = torch.nn.Module()
    model.mode = "lora_minilm_plus_projection"
    model.encoder = torch.nn.Linear(3, 3)
    model.lora_adapter = torch.nn.Linear(3, 3)
    model.projection = torch.nn.Linear(3, 3)
    set_phase(model, True)
    assert not model.lora_adapter.weight.requires_grad
    set_phase(model, False)
    x = torch.randn(2, 3)
    output = model.projection(model.encoder(x) + model.lora_adapter(x))
    output.square().mean().backward()
    assert model.lora_adapter.weight.grad is not None
    assert model.projection.weight.grad is not None
    assert model.encoder.weight.grad is None


@pytest.mark.parametrize("recipe", runner.RECIPES)
def test_actual_factory_freeze_contract_optimizer_and_gradients(monkeypatch, recipe):
    config = BertConfig(vocab_size=32, hidden_size=384, num_hidden_layers=3,
                        num_attention_heads=6, intermediate_size=32)
    monkeypatch.setattr(representation.AutoModel, "from_pretrained", lambda *a, **kw: BertModel(config))
    model = representation.build_racal_model(Path("unused-local-model"), recipe[1], 32)
    tokens = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
              "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]])}
    for warmup in (True, False):
        model.zero_grad(set_to_none=True)
        set_phase(model, warmup)
        for name, parameter in model.named_parameters():
            expected = name.startswith("projection.")
            if not warmup and recipe[0] == "last2_long":
                expected |= name.startswith(("encoder.encoder.layer.1.", "encoder.encoder.layer.2."))
            if not warmup and recipe[0] == "lora_long":
                expected |= "lora_" in name
            assert parameter.requires_grad == expected, name
        optimizer = optimizer_for(model, recipe[2])
        assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
            id(p) for p in model.parameters() if p.requires_grad}
        assert [g["lr"] for g in optimizer.param_groups] == (
            [2e-4] if warmup or recipe[0] == "projection_only" else [2e-4, recipe[2]])
        model(tokens)[:, 0].sum().backward()
        assert model.projection.fc2.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        if not warmup and recipe[0] == "lora_long":
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for n, p in model.named_parameters() if "lora_B" in n)


def save_completed(artifact, output, seed=13, recipe=runner.RECIPES[0], epochs=2):
    name, mode, lr = recipe
    run_dir = artifact / name / "clinc150" / f"kir50_seed{seed}" / "trainable_k1"
    run_dir.mkdir(parents=True, exist_ok=True)
    history = [dict(dataset="clinc150", seed=seed, strategy=name, mode=mode, split="val",
                    epoch=epoch, phase="warmup" if epoch == 1 else "finetune",
                    oos_f1=.7 if epoch == 1 else .8, center_count=1,
                    radius_lambda=1., threshold=1., acceptance_mode="nearest_sphere")
               for epoch in range(1, epochs + 2)]
    # Equal later scores must preserve the earliest winning epoch.
    result = dict(history[1], checkpoint=str(run_dir / "checkpoint.pt"), backbone_lr=lr,
                  projection_lr=2e-4, oos_used_for_training=False, test_used_for_selection=False,
                  checkpoint_objective="validation_oos_f1_only")
    runner.write_csv(output / f"{name}_seed{seed}_training_history.csv", history)
    runner.dump(run_dir / "run_manifest.json", result)
    torch.save(dict(model={"weight": torch.ones(1)}, mode=mode, epoch=2), run_dir / "checkpoint.pt")
    return result


@pytest.fixture
def resume_run(tmp_path):
    artifact, output = tmp_path / "artifacts", tmp_path / "output"
    artifact.mkdir()
    output.mkdir()
    manifest = dict(seeds=[13, 42, 87], recipes=runner.RECIPES, finetune_epochs=2,
                    artifact_root=str(artifact), status="running", direct_pipeline_verified=False)
    runner.dump(output / "MANIFEST.json", manifest)
    result = save_completed(artifact, output)
    empty = artifact / "projection_only/clinc150/kir50_seed13/trainable_k1"
    empty.mkdir(parents=True)
    return artifact, output, manifest, result


def test_resume_preserves_complete_checkpoint_history_and_allows_empty_failure(resume_run):
    artifact, output, manifest, result = resume_run
    files = [p for root in (artifact, output) for p in root.rglob("*") if p.is_file()]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
    assert runner.prepare_resume(artifact, output, manifest) == {(13, "last2_long"): result}
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}


@pytest.mark.parametrize("failure", ["missing_checkpoint", "partial_history", "history_only",
                                    "wrong_mode", "wrong_lr", "checkpoint_epoch", "wrong_winner",
                                    "changed_epochs", "test_result", "selected_artifact"])
def test_resume_rejects_partial_or_mismatched_results(resume_run, failure):
    artifact, output, manifest, result = resume_run
    checkpoint = Path(result["checkpoint"])
    history_path = output / "last2_long_seed13_training_history.csv"
    if failure == "missing_checkpoint":
        checkpoint.unlink()
    elif failure == "partial_history":
        history_path.write_text("\n".join(history_path.read_text().splitlines()[:-1]) + "\n")
    elif failure == "history_only":
        (output / "projection_only_seed13_training_history.csv").write_text("epoch\n1\n")
    elif failure in {"wrong_mode", "wrong_lr", "wrong_winner"}:
        result.update({"wrong_mode": {"mode": "trainable_projection_only"},
                       "wrong_lr": {"backbone_lr": .1}, "wrong_winner": {"epoch": 3}}[failure])
        runner.dump(checkpoint.parent / "run_manifest.json", result)
    elif failure == "checkpoint_epoch":
        torch.save(dict(model={"weight": torch.ones(1)}, mode=result["mode"], epoch=3), checkpoint)
    elif failure == "changed_epochs":
        manifest["finetune_epochs"] = 8
    elif failure == "test_result":
        (output / "selected_test_per_seed.csv").write_text("partial\n")
    elif failure == "selected_artifact":
        (artifact / "selected").mkdir()
    with pytest.raises(ValueError):
        runner.prepare_resume(artifact, output, manifest)


def test_training_refuses_to_overwrite_existing_history_before_model_load(resume_run, monkeypatch):
    artifact, output, _, _ = resume_run
    monkeypatch.setattr(runner, "build_racal_model", lambda *a: pytest.fail("must not load model"))
    with pytest.raises(ValueError, match="overwrite"):
        runner.train_recipe(13, runner.RECIPES[0], {}, torch.device("cpu"), artifact, output, 2)


def test_selection_lock_cannot_be_replaced_after_resume(tmp_path):
    lock = tmp_path / "lock.json"
    original = {"selected": [{"seed": 13, "epoch": 2}], "test_metrics_computed": False}
    runner.selection_lock(lock, original)
    runner.selection_lock(lock, original)
    with pytest.raises(ValueError, match="lock differs"):
        runner.selection_lock(lock, dict(original, selected=[{"seed": 13, "epoch": 3}]))
    assert json.loads(lock.read_text()) == original


def test_main_resumes_factory_failure_locks_every_seed_before_test_and_exports_metrics(tmp_path, monkeypatch):
    root, artifact, output = tmp_path / "s2c", tmp_path / "artifacts", tmp_path / "output"
    monkeypatch.setattr(runner, "ROOT", root)
    # No GPU, model training, or real data access in this orchestration regression.
    monkeypatch.setattr(runner.torch, "zeros", lambda *a, **kw: None)
    monkeypatch.setattr(runner.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(runner, "_RacalGateEncoder", lambda *a: SimpleNamespace(
        encode=lambda texts, **kw: np.zeros((len(texts), 1))))
    monkeypatch.setattr(runner, "validation_boundary", lambda train, val, views, seed, strategy: dict(
        dataset="clinc150", seed=seed, strategy=strategy, split="val", oos_f1=.7,
        center_count=1, radius_lambda=1., threshold=1., acceptance_mode="nearest_sphere"))
    known = {"text": "synthetic known", "intent": "known", "label": 0}
    unknown = {"text": "synthetic unseen", "intent": "unseen", "label": 1}
    for seed in (13, 42, 87):
        data = tmp_path / "assets/datasets/s2c/prepared/data/multidataset/v19/clinc150" / f"kir50_seed{seed}/gate"
        data.mkdir(parents=True)
        runner.dump(data / "train.json", [known])
        runner.dump(data / "val.json", [known, unknown])
        # Deliberately no test.json until the global lock is checked below.
    attempts = []
    fail_factory = True

    def fake_train(seed, recipe, views, device, artifact_root, output_root, epochs):
        assert set(views) == {"train", "val"}
        attempts.append((seed, recipe[0]))
        if fail_factory and recipe[0] == "projection_only":
            (artifact_root / "projection_only/clinc150/kir50_seed13/trainable_k1").mkdir(parents=True)
            raise ValueError("Unsupported RACAL representation mode")
        return save_completed(artifact_root, output_root, seed, recipe, epochs)

    monkeypatch.setattr(runner, "train_recipe", fake_train)
    test_seeds = []

    def encode_test(dataset, seed, device, selected_root):
        global_lock = json.loads((output / "all_seeds_selection_lock.json").read_text())
        assert [r["seed"] for r in global_lock["selected"]] == [13, 42, 87]
        assert global_lock["test_metrics_computed"] is False
        for locked_seed in (13, 42, 87):
            lock = json.loads((output / f"seed{locked_seed}_selection_lock.json").read_text())
            assert len(lock["candidates"]) == 4
            assert lock["selected"]["epoch"] == 2
            assert lock["test_metrics_computed"] is False
        test_seeds.append(seed)
        return ({"train": [known], "val": [known, known, unknown, unknown], "test": [known, known, unknown, unknown]},
                {"train": np.array([.2]), "val": np.array([.2, 1.2, .8, 1.4]), "test": np.array([.2, 1.2, .8, 1.4])})

    monkeypatch.setattr(runner, "_encode_cell", encode_test)
    monkeypatch.setattr(runner, "_fit_detector", lambda *a: SimpleNamespace(
        acceptance_mode="nearest_sphere", cluster_to_intent={0: "known"}))
    monkeypatch.setattr(runner, "_distance_matrix", lambda d, values: values)
    monkeypatch.setattr(runner, "_score_from_distances", lambda values, *a: dict(
        score=values, nearest_cluster=np.zeros(len(values), dtype=int)))
    monkeypatch.setattr(runner, "_run_pipeline", lambda *a: ([
        {"intent": "wrong" if score == .2 else "known", "is_oos": score > 1., "gate_score": score}
        for score in (.2, 1.2, .8, 1.4)], {"prediction_count": 4}))
    argv = ["runner", "--epochs", "2", "--artifact-root", str(artifact), "--output-root", str(output)]
    monkeypatch.setattr(runner.sys, "argv", argv)
    with pytest.raises(ValueError, match="Unsupported RACAL"):
        runner.main()
    assert test_seeds == []
    preserved = [artifact / "last2_long/clinc150/kir50_seed13/trainable_k1/checkpoint.pt",
                 artifact / "last2_long/clinc150/kir50_seed13/trainable_k1/run_manifest.json",
                 output / "last2_long_seed13_training_history.csv"]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in preserved}
    fail_factory = False
    monkeypatch.setattr(runner.sys, "argv", [*argv, "--resume"])
    runner.main()
    assert attempts.count((13, "last2_long")) == 1
    assert test_seeds == [13, 42, 87]
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in preserved}
    with (output / "selected_test_per_seed.csv").open() as stream:
        selected = list(csv.DictReader(stream))
    with (output / "selected_test_summary.csv").open() as stream:
        summary = next(csv.DictReader(stream))
    assert [float(r["gate_f1_all"]) for r in selected] == pytest.approx([.5] * 3)
    assert [float(r["full_pipeline_f1_all"]) for r in selected] == pytest.approx([.25] * 3)
    assert float(summary["gate_f1_all_mean"]) == pytest.approx(.5)
    assert float(summary["full_pipeline_f1_all_mean"]) == pytest.approx(.25)
    assert float(summary["full_pipeline_f1_all_std"]) == 0.
    assert all(sum(int(v) for k, v in row.items() if k.startswith("full_pipeline_stage_")) == 4
               for row in selected)
    with (output / "ranking_diagnostics.csv").open() as stream:
        diagnostics = list(csv.DictReader(stream))
    assert {(int(r["seed"]), r["split"]) for r in diagnostics} == {
        (seed, split) for seed in (13, 42, 87) for split in ("val", "test")}
    assert all(r["oracle_is_posthoc_not_selection"] == "True" for r in diagnostics)
    assert all(float(r["fixed_score_threshold_oracle_f1"]) == pytest.approx(.8) for r in diagnostics)
    with (output / "intent_errors.csv").open() as stream:
        errors = list(csv.DictReader(stream))
    assert len(errors) == 12
    assert all(int(r["accepted_count"]) + int(r["rejected_count"]) == int(r["count"]) for r in errors)
    assert all("text" not in r for r in errors)

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import yaml

from scripts.experiments.run_mogb_official_modernized_smoke import (
    DEFAULT_CONFIG_PATH,
    file_or_tree_hash,
    main as smoke_main,
)
from tests.unit.test_mogb_legacy_compat import (
    MOGB_OFFICIAL_ROOT,
    _legacy_import_paths,
    _write_local_bert_assets,
)


def _load_official_model(tmp_path: Path, num_labels: int = 3):
    model_dir = tmp_path / "tiny-bert"
    config = _write_local_bert_assets(model_dir)
    with _legacy_import_paths():
        import importlib

        modeling = importlib.import_module("pytorch_pretrained_bert.modeling")
        tokenization = importlib.import_module("pytorch_pretrained_bert.tokenization")
        official_model = importlib.import_module("model")

        reference = official_model.BertForModel(config, num_labels=num_labels)
        torch.save(reference.state_dict(), model_dir / modeling.WEIGHTS_NAME)
        config.to_json_file(model_dir / modeling.CONFIG_NAME)

        tokenizer = tokenization.BertTokenizer.from_pretrained(str(model_dir), do_lower_case=True)
        encoded = tokenizer(
            [
                "hello world",
                "tiny bert",
                "hello tiny world",
                "bert world",
                "tiny hello",
            ],
            padding="max_length",
            truncation=True,
            max_length=6,
            return_tensors="pt",
        )
        model = official_model.BertForModel.from_pretrained(str(model_dir), num_labels=num_labels)
    return model, encoded


def _gradients(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            output[name] = parameter.grad.detach().clone()
    return output


def test_weighted_chunk_losses_match_full_batch_mean_gradient(tmp_path: Path) -> None:
    model, encoded = _load_official_model(tmp_path)
    labels = torch.tensor([0, 1, 2, 1, 0])
    full_model = copy.deepcopy(model)
    chunked_model = copy.deepcopy(model)
    total = labels.numel()

    full_model.eval()
    full_model.zero_grad(set_to_none=True)
    full_loss = full_model(
        encoded["input_ids"],
        encoded["token_type_ids"],
        encoded["attention_mask"],
        labels=labels,
        mode="train",
    )
    full_loss.backward()
    full_gradients = _gradients(full_model)

    chunked_model.eval()
    chunked_model.zero_grad(set_to_none=True)
    for chunk in (slice(0, 2), slice(2, 5)):
        batch_loss = chunked_model(
            encoded["input_ids"][chunk],
            encoded["token_type_ids"][chunk],
            encoded["attention_mask"][chunk],
            labels=labels[chunk],
            mode="train",
        )
        weighted = batch_loss * (labels[chunk].numel() / total)
        weighted.backward()
    chunked_gradients = _gradients(chunked_model)

    assert full_gradients.keys() == chunked_gradients.keys()
    for name in full_gradients:
        assert torch.allclose(full_gradients[name], chunked_gradients[name], atol=1e-6, rtol=1e-6), name


def test_detached_granular_ball_features_drop_autograd_history(tmp_path: Path) -> None:
    model, encoded = _load_official_model(tmp_path)
    model.train()

    features = model(
        encoded["input_ids"][:3],
        encoded["token_type_ids"][:3],
        encoded["attention_mask"][:3],
        feature_ext=True,
    )
    detached = features.detach().cpu()

    assert features.requires_grad
    assert features.grad_fn is not None
    assert not detached.requires_grad
    assert detached.grad_fn is None


def test_modernized_launcher_dry_run_keeps_official_checkout_hash_stable(tmp_path: Path, capsys) -> None:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["output_root"] = str(tmp_path / "mogb-official-smoke")
    config_path = tmp_path / "mogb_official_modernized_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    before = file_or_tree_hash(MOGB_OFFICIAL_ROOT)["sha256"]
    exit_code = smoke_main(
        [
            "--config",
            str(config_path),
            "--dataset",
            "stackoverflow",
            "--kir",
            "0.50",
            "--seed",
            "0",
            "--dry-run",
        ]
    )
    after = file_or_tree_hash(MOGB_OFFICIAL_ROOT)["sha256"]

    assert exit_code == 0
    assert before == after

    payload = json.loads(capsys.readouterr().out.strip())
    manifest = Path(payload["manifest"])
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "dry_run"
    assert saved["status"] == "dry_run"
    assert saved["launch_ready"] is True

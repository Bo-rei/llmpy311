from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import BertConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGB_OFFICIAL_ROOT = PROJECT_ROOT / "third_party" / "mogb_official"
MOGB_COMPAT_ROOT = PROJECT_ROOT / "third_party" / "mogb_compat"


def _clear_legacy_modules() -> None:
    prefixes = ("pytorch_pretrained_bert", "utils")
    exact = {"dataloader", "model", "pretrain", "util"}
    for name in list(sys.modules):
        if name in exact or name.startswith(prefixes):
            sys.modules.pop(name, None)


@contextmanager
def _legacy_import_paths():
    _clear_legacy_modules()
    original_path = list(sys.path)
    sys.path.insert(0, str(MOGB_OFFICIAL_ROOT))
    sys.path.insert(0, str(MOGB_COMPAT_ROOT))
    try:
        yield
    finally:
        sys.path[:] = original_path
        _clear_legacy_modules()


def _write_local_bert_assets(model_dir: Path) -> BertConfig:
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "tiny", "bert"]
    (model_dir / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    return config


def test_official_imports_resolve_through_compat_layer() -> None:
    with _legacy_import_paths():
        modeling = importlib.import_module("pytorch_pretrained_bert.modeling")
        tokenization = importlib.import_module("pytorch_pretrained_bert.tokenization")
        optimization = importlib.import_module("pytorch_pretrained_bert.optimization")
        compat_util = importlib.import_module("utils.util")
        official_pretrain = importlib.import_module("pretrain")
        assert modeling.WEIGHTS_NAME == "pytorch_model.bin"
        assert modeling.CONFIG_NAME == "config.json"
        assert tokenization.BertTokenizer is not None
        assert optimization.BertAdam is not None
        assert hasattr(official_pretrain, "PretrainModelManager")
        assert callable(compat_util.F_measure)
        compat_util.summary_writer.add_scalar("loss/train", 1.0, 0)
        compat_util.summary_writer.close()


def test_bert_wrapper_preserves_legacy_forward_contract(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-bert"
    config = _write_local_bert_assets(model_dir)

    with _legacy_import_paths():
        modeling = importlib.import_module("pytorch_pretrained_bert.modeling")
        tokenization = importlib.import_module("pytorch_pretrained_bert.tokenization")
        official_model = importlib.import_module("model")

        reference = official_model.BertForModel(config, num_labels=3)
        torch.save(reference.state_dict(), model_dir / modeling.WEIGHTS_NAME)
        config.to_json_file(model_dir / modeling.CONFIG_NAME)

        tokenizer = tokenization.BertTokenizer.from_pretrained(str(model_dir), do_lower_case=True)
        encoded = tokenizer(
            ["hello world", "tiny bert"],
            padding="max_length",
            truncation=True,
            max_length=6,
            return_tensors="pt",
        )

        loaded = official_model.BertForModel.from_pretrained(str(model_dir), num_labels=3)
        loaded.eval()

        encoder_layers, pooled_output = loaded.bert(
            encoded["input_ids"],
            encoded["token_type_ids"],
            encoded["attention_mask"],
            output_all_encoded_layers=True,
        )
        sequence_output, pooled_output_single = loaded.bert(
            encoded["input_ids"],
            encoded["token_type_ids"],
            encoded["attention_mask"],
            output_all_encoded_layers=False,
        )
        features, logits = loaded(
            encoded["input_ids"],
            encoded["token_type_ids"],
            encoded["attention_mask"],
            mode="eval",
        )
        loss = loaded(
            encoded["input_ids"],
            encoded["token_type_ids"],
            encoded["attention_mask"],
            labels=torch.tensor([0, 1]),
            mode="train",
        )

    assert len(encoder_layers) == config.num_hidden_layers
    assert encoder_layers[-1].shape == (2, 6, config.hidden_size)
    assert pooled_output.shape == (2, config.hidden_size)
    assert sequence_output.shape == (2, 6, config.hidden_size)
    assert pooled_output_single.shape == (2, config.hidden_size)
    assert features.shape == (2, 768)
    assert logits.shape == (2, 3)
    assert loss.ndim == 0


def test_bert_adam_approximates_legacy_warmup_schedule() -> None:
    with _legacy_import_paths():
        optimization = importlib.import_module("pytorch_pretrained_bert.optimization")

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = optimization.BertAdam(
        [parameter],
        lr=0.01,
        warmup=0.4,
        t_total=10,
        max_grad_norm=0.1,
    )

    lrs = []
    for _ in range(6):
        parameter.grad = torch.tensor([5.0])
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad()

    assert parameter.item() != 1.0
    assert all(lr >= 0.0 for lr in lrs)
    assert lrs[1] > lrs[0]
    assert lrs[-1] < max(lrs)

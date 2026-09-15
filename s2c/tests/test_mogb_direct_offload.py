import io
import copy
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.experiments.run_mogb_shared_official import (
    CpuSavedTensor,
    CpuFp16SavedTensor,
    CpuInt8SavedTensor,
    DirectTensorStore,
    _direct_read,
    _direct_write,
    _feature_only_saved_tensors,
)


def test_direct_saved_tensor_storage_roundtrip():
    value = np.arange(131071, dtype=np.float32).reshape(-1, 1)
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tensor.npy"
        payload_size, padded_size = _direct_write(path, stream.getvalue())
        assert padded_size % 4096 == 0
        restored = np.load(
            io.BytesIO(_direct_read(path, payload_size, padded_size)),
            allow_pickle=False,
        )

    np.testing.assert_array_equal(restored, value)


def test_direct_tensor_store_roundtrip_and_single_backing_file():
    values = [np.arange(12345, dtype=np.float32), np.arange(6789, dtype=np.int64)]
    with tempfile.TemporaryDirectory() as directory:
        with DirectTensorStore(directory) as store:
            records = []
            for value in values:
                stream = io.BytesIO()
                np.save(stream, value, allow_pickle=False)
                records.append((value, store.write(stream.getvalue())))
            backing_files = list(Path(directory).glob('*.bin'))
            assert len(backing_files) == 1
            for expected, (offset, payload_size, padded_size) in records:
                payload = store.read(offset, payload_size, padded_size)
                restored = np.load(io.BytesIO(payload), allow_pickle=False)
                np.testing.assert_array_equal(restored, expected)


def test_direct_tensor_store_resets_between_epochs():
    value = np.arange(8193, dtype=np.float32)
    with tempfile.TemporaryDirectory() as directory:
        with DirectTensorStore(directory) as store:
            stream = io.BytesIO()
            np.save(stream, value, allow_pickle=False)
            first = store.write(stream.getvalue())
            assert store.offset > 0
            store.reset_epoch()
            assert store.offset == 0
            assert next(Path(directory).glob('*.bin')).stat().st_size == 0
            second = store.write(stream.getvalue())
            restored = np.load(
                io.BytesIO(store.read(*second)), allow_pickle=False
            )
            np.testing.assert_array_equal(restored, value)
            assert first[0] == 0 and second[0] == 0


def test_feature_only_cpu_hook_preserves_backward_and_restores_forward():
    class Toy(nn.Module):
        def forward(self, value, feature_ext=False, mode=None):
            hidden = value * 2
            if feature_ext:
                return hidden.square()
            return hidden.square().sum()

    model = Toy()
    original_forward = type(model).forward
    value = torch.tensor([1.5], requires_grad=True)
    with _feature_only_saved_tensors(model, CpuSavedTensor):
        copy.deepcopy(model)
        model(value, mode='train').backward()
        model(value, feature_ext='True').sum().backward()
    assert type(model).forward is original_forward
    assert value.grad is not None


def test_cpu_fp16_saved_tensor_roundtrip_restores_dtype():
    value = torch.arange(4096, dtype=torch.float32, requires_grad=True)
    saved = CpuFp16SavedTensor(value)
    assert saved.compressed
    assert saved.value.dtype == torch.float16
    restored = saved.unpack()
    assert restored.dtype == value.dtype
    assert restored.device == value.device
    assert torch.allclose(restored, value, atol=1e-2, rtol=1e-3)


def test_cpu_int8_saved_tensor_roundtrip_restores_dtype():
    value = torch.linspace(-3.0, 3.0, 4096, dtype=torch.float32, requires_grad=True)
    saved = CpuInt8SavedTensor(value)
    assert saved.compressed
    assert saved.value.dtype == torch.int8
    restored = saved.unpack()
    assert restored.dtype == value.dtype
    assert restored.device == value.device
    assert torch.allclose(restored, value, atol=3.0 / 127.0, rtol=1e-3)

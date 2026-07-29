from pathlib import Path

from protocol_v2.data.hashing import atomic_write_jsonl, sha256_file, sha256_json


def test_stable_hashing_and_atomic_jsonl(tmp_path: Path) -> None:
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})
    path = tmp_path / "records.jsonl"
    atomic_write_jsonl(path, [{"b": 2, "a": 1}])
    first = sha256_file(path)
    atomic_write_jsonl(path, [{"a": 1, "b": 2}])
    assert sha256_file(path) == first


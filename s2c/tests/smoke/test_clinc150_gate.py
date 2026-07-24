from ._gate_inputs import assert_local_gate_inputs


def test_clinc150_gate_inputs_are_local_and_complete() -> None:
    assert_local_gate_inputs("clinc150")

from protocol_v2.data.schema import ALL_REGISTRY_SEEDS, DATASET_SPECS, FORMAL_KIRS, format_kir


def test_protocol_v2_dataset_and_registry_constants_are_fixed() -> None:
    assert tuple(DATASET_SPECS) == ("clinc150", "banking77", "stackoverflow")
    assert len(FORMAL_KIRS) == 11
    assert len(ALL_REGISTRY_SEEDS) == 13
    assert format_kir(0.5) == "0.50"


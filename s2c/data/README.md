# s2c local data layer

`s2c/data/` is the only runtime data root for protocol_v2. The source snapshot
is copied byte-for-byte from the fixed local TEXTOIR commit during import;
TEXTOIR is never a runtime data dependency after that point.

Raw source files, canonical JSONL, views, exports, caches and temporary files
are deliberately ignored by Git. Small manifests and KIR registries are kept
as provenance records. Build the layer in this order:

```bash
python -m s2c.data.import_textoir
python -m s2c.data.build_canonical
python -m s2c.data.build_registries
python -m s2c.data.build_views
python -m s2c.data.validate_protocol
```

Do not substitute `../assets/datasets`, `../textoir/data`, network downloads,
the historical Banking77-OOS extension or the old deduplicated StackOverflow
snapshot. The exact source and licensing decisions are documented in
[`docs/audits/data_provenance/`](../docs/audits/data_provenance/).


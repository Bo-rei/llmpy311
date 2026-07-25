# TEXTOIR runtime independence

Status: **passed**. The check temporarily renamed `../textoir` to `textoir.disabled`, ran full canonical/registry/view/export validation, Gate dry-run and Gate data loading, then restored the directory and checked its Git status. No model training or embedding generation was used for this check.

# ADB / DA-ADB baseline audit

## Scope and current status

This document records runnability evidence for the two boundary-learning
baselines requested alongside MOGB. It is not a result table: no ADB or
DA-ADB metric is claimed for the current `protocol_v2_textoir_v1` protocol.

| Method | Source entrypoint | Current status | Reason no metric is reported |
| --- | --- | --- | --- |
| ADB | `tools/compat/textoir/run_external_textoir.py` -> `methods/ADB/manager.py` | `complete_compatibility_artifact` | StackOverflow/KIR=.50/seed=0 completed in an isolated `textoir-py39` overlay; this is modernized compatibility evidence, not strict paper reproduction. |
| DA-ADB | same manager with `configs/DA-ADB.py` and `bert_disaware` | `complete_compatibility_artifact` | StackOverflow/KIR=.50/seed=0 completed under the same isolated overlay and local BERT conversion contract. |

ADB and DA-ADB remain independent boundary baselines. They are not a
replacement for DCLOOS, whose end-to-end pseudo-OOS plus external-OOS
supervision contract is separately blocked by missing `squad_placeh.tsv`.

## Source and adapter audit

* TEXTOIR source revision: `dffe2b1b848a069a6808f8089b4cb9bd16e2062b`.
* ADB config SHA256: `620250896056a49b0b8902faa9bfb38c99f55e4bf32ec98be925e5cc4fa112b1`.
* DA-ADB config SHA256: `cd58b4fd0df6300725af28c65da5323ad68c0a8d06745f5e0c76b319a13938fa`.
* Shared ADB manager SHA256:
  `8c9403784cdf56765878f363cd920164b98df71af3b09e129db495825306b3d3`.
* ADB boundary SHA256:
  `4079152ec2a426ec6062c6507ea862ed3e246e22e7bbcec31929fa74ffdeab54`.
* Adapter: `tools/compat/textoir/run_external_textoir.py` creates an isolated
  overlay, patches only the local BERT path, optional imports, missing BERT
  dataloader routes, and ADB CUDA diagnostic serialization. The TEXTOIR
  checkout remains read-only and clean.

The upstream method map registers both `ADB` and `DA-ADB` to
`ADBManager`; the distinction is the configuration and `bert_disaware`
backbone route. The two completed rows below are therefore separate
modernized compatibility cells, while their shared implementation and
single-seed scope remain explicit.

## Data and command contract

The wrapper's dry-run confirms the intended legacy commands for a
StackOverflow/KIR=0.50/seed=0 smoke:

* ADB: `bert`, `configs/ADB.py`, `--pretrain`, `--save_model`;
* DA-ADB: `bert_disaware`, `configs/DA-ADB.py`, `--pretrain`.

The upstream command still points at the legacy TEXTOIR TSV root. A fair
protocol run would require a fixed adapter to the same canonical registry,
Known intent list and train/dev/test rows used by s2c. The existing
`protocol_v2` ADB/DA-ADB exports are input-format exports only; they do not
constitute method reproduction.

## Preflight evidence

The initial isolated overlay probe was executed for both methods. Both failed
before method registration because the installed Transformers version removed
the legacy top-level import:

```text
ImportError: cannot import name 'AdamW' from 'transformers'
```

No training, checkpoint, threshold selection, or test evaluation was started.
This is a deterministic environment blocker rather than a performance result.
The dry-run manifests under `/tmp/s2c_ADB_dryrun` and
`/tmp/s2c_DA_ADB_dryrun` are disposable probes and are not project evidence.

The compatibility retry changes only the copied overlay: `backbones/base.py`
uses `torch.optim.AdamW` while accepting the legacy `correct_bias` keyword;
the pinned TEXTOIR checkout remains untouched. Because torch's optimizer
semantics are not byte-identical to the removed Hugging Face implementation,
any resulting metric will be labeled modernized compatibility, not strict
paper reproduction.

## Re-entry conditions

To run either baseline later, register a new experiment rather than changing
this audit. The new run must:

1. use an isolated legacy/modern compatibility environment with an explicit
   optimizer import;
2. preserve the method's Known-only training and validation contract;
3. consume the same fixed registry and split manifest as the s2c controls;
4. record the overlay patch, environment, data hashes and metric schema; and
5. keep ADB/DA-ADB rows separate from the DCLOOS end-to-end row.

The completed compatibility retries are recorded in
`../artifacts/s2c/external/adb_compat_single_cell_v2/` and
`../artifacts/s2c/external/da_adb_compat_single_cell_v3/`; they do not overwrite
the historical v19 artifact or claim strict protocol_v2 reproduction.

## Existing ADB compatibility artifact

An earlier isolated TEXTOIR run is already complete at
`../artifacts/s2c/outputs/experiments/cluster_separability_v19/textoir_protocol/official_runs/stackoverflow/ADB/kir50/seed0/attempts/attempt_0001/`.
Its manifest records a clean upstream checkout, a complete prediction audit,
the BERT model environment and an unchanged runtime overlay.  The single-cell
result is Accuracy `88.37`, F1-known `87.2645`, F1-open `89.3146`, and overall
F1 `87.4508`.  This is retained as historical TEXTOIR compatibility evidence;
it is not relabeled as a `protocol_v2_textoir_v1` strict reproduction because
the artifact belongs to the frozen v19 official-run contract.

## Current compatibility single cells

Both runs use the same StackOverflow TEXTOIR train/dev/test snapshot, KIR=.50,
seed=0, local `bert-base-uncased` weights converted from the ignored
`model.safetensors` file to an isolated `pytorch_model.bin`, and a torch-native
AdamW overlay. The upstream TEXTOIR checkout remained clean.

| Method | Accuracy | F1-known | F1-open | F1-All | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| ADB | 88.53 | 87.4428 | 89.4712 | 87.6272 | `../artifacts/s2c/external/adb_compat_single_cell_v2/stackoverflow/ADB/kir50/seed0/` |
| DA-ADB | 90.07 | 89.0584 | 90.8978 | 89.2256 | `../artifacts/s2c/external/da_adb_compat_single_cell_v3/stackoverflow/DA-ADB/kir50/seed0/` |

These are single-seed modernized compatibility results. They are useful
external boundary references, but are not five-seed unified-registry results
and must not be called strict paper reproduction or direct SOTA evidence.

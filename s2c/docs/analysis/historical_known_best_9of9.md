# Known-only OOS-F1 progress report

The report joins two CUDA Gate-only final-evaluation campaigns under the
historical v19 data family. Training and validation rows contain Known intents
only; no real OOS or pseudo-OOS row is used for selection. The three seeds are
13, 42 and 87. The external comparison values are the strongest registered
OOS-F1 references in the current evaluation runner.

The CLINC150 and BANKING77-OOS rows use the fixed Known-validation coverage
90% selection campaign. The StackOverflow rows use a fixed Known-only
standard contract: K=1, Euclidean distance, mean+1.5 std radius,
normalized-union acceptance, no score fusion, and threshold 0.90. The raw
summary and per-seed values are in the same result directory.

Numerically, all nine rows exceed their registered reference OOS F1. This is
exploratory progress rather than a clean untouched-holdout claim: earlier
exploratory test artifacts existed in the workspace before the final
StackOverflow contract was fixed. A paper claim should use a fresh locked
rerun or disclose this protocol history.

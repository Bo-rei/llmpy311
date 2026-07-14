# LLMPY311 workspace

This is a multi-project research workspace, not one Python package. Its
top-level directories have one responsibility each:

- `s2c/` — active HiLSA-MoE v19 codebase. Start at its
  [documentation index](./s2c/docs/README.md).
- `assets/models/` and `assets/datasets/` — shared local models and small
  standalone datasets.
- `archives/` — legacy research, idea-search output, submissions, and dated source snapshots.

## Migration plan and compatibility

There are no root-level compatibility symlinks. Use `projects/s2c` for code,
`assets` for models and datasets, `artifacts` for generated results, and
`archives` for non-active material.

The migration does not delete models, datasets, experiment outputs, or the
submission/snapshot archives. `s2c_submission` remains whole under
`archives/submissions/` because it is a self-contained reproduction bundle,
not a source tree to merge into `s2c`.

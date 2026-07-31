# Official MOGB code audit

Source: `https://github.com/Liyanhuaa/MOGB.git`  
Pinned commit: `5b689e2a03de0d86ec41212825e5db8d7f0e5c02`  
Acquired: 2026-07-29  
Local checkout: `third_party/mogb_official/`

The checkout is retained as third-party source.  It is not copied into the
active `protocol_v2` package and its algorithm is not claimed as an s2c
contribution.

## Observed flow

1. `pretrain.py` loads BERT and trains a classification head on Known data.
2. The last encoder layer and pooler remain trainable when `freeze_bert` is set.
3. At each training epoch, the code creates features and applies the nearest
   sub-centroid loss (`myloss.py`).
4. `cluster3.py` recursively splits a mixed ball using one randomly selected
   point per other label, then assigns points to the nearest split seed.
5. Selected balls are filtered by purity and minimum sample count.
6. A ball center is the mean feature; the active radius is the **mean** Euclidean
   distance to that center (`calculate_center_and_radius`).
7. `gb_test.py` assigns a test point to the nearest selected center and accepts
   it only when its distance is strictly smaller than that ball radius.

## Compatibility findings

* `run.sh` has a malformed shebang and shell control tokens (`do/done`, `seed 0\`).
* The checkout imports a `utils` package that is not present in the repository.
* `requirements.txt` pins the obsolete PyTorch 1.7 and
  `pytorch-pretrained-bert` stack.
* Several paths hard-code `cuda:0`; CPU and arbitrary GPU execution are not
  safe without compatibility edits.
* Purity arguments are declared as `int` in `init_parameter.py` although the
  defaults are fractional.
* The official dataloader regenerates Known labels and reads its own legacy
  file format, so it cannot be used for a fair protocol_v2 comparison.
* No `LICENSE` file was found in the pinned checkout.  The source is therefore
  kept as an audit/reference checkout and is not redistributed as project code.

## Reproduction status

The original BERT/TextOIR run is **audited but not yet reproduced**.  A fair
MiniLM mode is runnable from cached protocol_v2 embeddings.  This distinction
is required: a MOGB partition adapter on MiniLM is not the authors' full
representation-learning method.

## Explicit reproduction assumptions

The active adapter uses the official final-ball semantics (purity-driven
recursive split, minimum selected-ball size, majority label, mean radius,
nearest-ball open decision) but uses integer-indexed arrays and the fixed
protocol_v2 registry.  It does not claim byte-for-byte equivalence with the
legacy autograd implementation until an original-contract run is available.

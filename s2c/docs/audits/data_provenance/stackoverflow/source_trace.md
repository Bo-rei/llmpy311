# StackOverflow source and license trace

## Verified content reference

- Repository: `https://github.com/jacoxu/StackOverflow`
- Fixed revision: `7c207f51e649fff9e4736610b9d44431bb7ccf00`
- Raw files: `rawText/title_StackOverflow.txt` and `rawText/label_StackOverflow.txt`
- File SHA256: `56fdb…22e86` and `ac6b…d7d3` respectively
- README SHA256: `3d51882c2a3ec462309fb04afa0f3039858881303ffa577d3cf300e1e8d72b4d`

The README identifies the corpus as 20,000 StackOverflow titles over 20 labels and asks users to
cite the 2015 NAACL VSM-NLP workshop paper and acknowledge Kaggle. It does not name a Kaggle dataset,
data-dump revision, source post identifiers, collection date, transformation rule, or license. The
repository has no `LICENSE` file at the fixed revision (the raw `LICENSE` endpoint returns HTTP 404).

## Why the general Stack Overflow policy is insufficient

Stack Overflow states that the applicable CC BY-SA version varies with contribution date and that the
license for an individual revision is available on that post's timeline. The 20,000-row corpus contains
only title text and a derived label: it has no post ID, author, date, revision, or data-dump provenance.
Therefore neither the applicable license version nor attribution chain can be verified for every row.

## Decision

This establishes a reproducible **content reference**, not a canonical raw source. The dataset remains
`blocked_unverified`; it must not enter `protocol_v2_official_v1`, formal training, embedding generation,
external-method reproduction, or TEXTOIR-fair-comparability claims. The precise decision is in
`dataset_decision.json`.

## Sources

- [Upstream dataset README](https://raw.githubusercontent.com/jacoxu/StackOverflow/7c207f51e649fff9e4736610b9d44431bb7ccf00/README.md)
- [Stack Overflow licensing help](https://stackoverflow.com/help/licensing)
- [Stack Exchange data-dump announcement](https://stackoverflow.blog/2014/01/23/stack-exchange-cc-data-now-hosted-by-the-internet-archive/)

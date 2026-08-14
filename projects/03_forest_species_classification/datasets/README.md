# Dataset layout

The BarkVN-50 image payload is intentionally not committed.

Download Version 1 from https://data.mendeley.com/datasets/gbt4tdmttn/1 and extract the images to:

```text
datasets/raw/BarkVN-50/v1/images/BarkVN-50_mendeley/
```

`processed/` contains the small reproducibility metadata produced by the audit and grouped-split stages:

- `class_to_idx.json`
- `exclusions.csv`
- `similarity_groups.csv`
- `split_manifest.csv`

See `docs/data_source.md` and `docs/data_quality_review.md` for provenance and limitations.

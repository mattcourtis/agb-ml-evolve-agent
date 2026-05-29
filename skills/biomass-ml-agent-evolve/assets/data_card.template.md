---
license: "Specify"
language:
- en
tags:
- biomass
- forest-structure
- remote-sensing
- lidar
pretty_name: "{experiment_id} dataset"
size_categories:
- "Specify"
---

# Data Card

## Dataset summary

One paragraph describing what the dataset contains, the subject, the geography, the temporal window, and the intended use (training, validation, or test for the {task_type} task on {subject} in {geography}).

## Sources

| source_name | url_or_doi | access_date | license |
|---|---|---|---|
| TBD | TBD | TBD | TBD |

## Collection window

- start_date: TBD
- end_date: TBD
- label_source_revision_tag: TBD

## Geography

- covered_regions: TBD
- exclusions: TBD

## Splits

Must match the ACCEPTED `configs/split_strategy.yaml`.

| split | n_units | partition_key | partition_value_count |
|---|---:|---|---:|
| train | TBD | TBD | TBD |
| val   | TBD | TBD | TBD |
| test  | TBD | TBD | TBD |

## Label generation

How labels were derived, what produced them, and known biases (e.g., self-reported, satellite-inferred, crowdsourced, official statistic revised retroactively).

For biomass tasks: use field-measured plot labels only (ANEW, USFS FIA, NEON woody veg). **Do not use GEDI L4A/L4B gridded biomass or ESA CCI Biomass products as training labels** — they are benchmarks / cross-validation references, not independent ground truth; treating them as labels is circular and inflates apparent performance against the same product family.

## Known limitations and biases

- TBD

## Intended use

- Primary use: TBD
- Out-of-scope use: TBD

## Citation

```
TBD
```

## Versioning

- content_sha256: TBD   # must equal the SHA256 recorded in preprocessing/data_version.txt
- snapshot_timestamp_utc: TBD
- input_manifest_path: TBD

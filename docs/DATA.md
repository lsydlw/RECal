# Data preparation

## Required layout

Set `DATA_ROOT` to a directory with the following layout:

```text
data/
├── ml-100k/
├── datafoundation-user/
├── Bookmarks/
├── LastFM/
├── amazon_music/
├── foursquare/
├── online_retail/
└── baidu_news_v2/
```

For compatibility with the original server snapshot, the four core folders
may instead be named `ml-100k_v2`, `datafoundation-user_v2`, `Bookmarks_v2`,
and `LastFM_v2`. The portable scripts detect either name.

Every dataset directory must contain exactly these inputs:

| File | Required fields | Meaning |
|---|---|---|
| `train.csv` | `user_id,item_id,timestamp` | Chronological training interactions |
| `valid.csv` | `user_id,item_id,timestamp` | Validation targets/interactions |
| `test.csv` | `user_id,item_id,timestamp` | Held-out test targets/interactions |
| `item_meta.csv` | `item_id` plus a category field | Item-to-category mapping |

Recognized category field names are `category`, `item_category`, `cate_id`,
`genre`, `beer_style`, `cluster_id`, and `category_id`. IDs must be numeric.
Within each user, rows must be ordered by `(timestamp, item_id)`, and the split
chronology must be train before validation before test.

The evaluator performs full-catalog ranking over every non-padding item and
keeps history items in the candidate set. Validation NDCG@10 selects the
checkpoint. Test is read only after the manifest has frozen all selected runs.

## Validate processed data

```bash
python3 tools/validate_data.py data/ml-100k
```

Run the same command for all eight directories before launching experiments.

## Installation-only toy data

The toy generator is only a smoke test; it is not a paper dataset:

```bash
python3 tools/make_toy_data.py --output data/toy
python3 tools/validate_data.py data/toy
```

## Redistribution

The original datasets have different licenses and redistribution conditions.
This repository therefore distributes the preprocessing contract and code,
not the raw data. Obtain each source dataset under its own license, create the
four files above, and retain the source/license metadata in your local data
directory. The private archive records the already-run experiment artifacts
but deliberately excludes the 189 GB server data tree.


# Digg GraphSAGE preflight

## 1. Environment

- torch: `2.14.0`
- torch_geometric: `2.8.0.post1`
- `from torch_geometric.nn import SAGEConv`: PASS (`SAGEConv`)

## 2. Required data files

- PASS: `data/processed/digg_friends_clean.csv.gz` (14,092,155 bytes)
- PASS: `data/processed/digg_communities.csv` (2,263,912 bytes)
- PASS: `data/processed/digg_exposure_pilot.csv.gz` (2,853,957 bytes)
- PASS: `outputs/digg/pilot_story_split.csv` (3,297 bytes)
- Auxiliary temporal input: `data/processed/digg_votes_clean.csv.gz` (20,127,550 bytes)

## 3. Exposure schema

- Shape: `(346338, 11)`
- Columns: `['story_id', 'time_bin', 'node_id', 'community', 'm_in', 'm_out', 'degree', 'frac_in', 'frac_out', 'y', 'sampling_weight']`

First five rows (`pandas.DataFrame.head()`):

```text
 story_id  time_bin  node_id  community  m_in  m_out  degree  frac_in  frac_out  y  sampling_weight
       46         0    75716          2     0      0     126      0.0       0.0  1              1.0
       46         0    90782          2     0      0     153      0.0       0.0  1              1.0
       46         0   110683         16     0      0       4      0.0       0.0  1              1.0
       46         0   112656          2     0      0      71      0.0       0.0  1              1.0
       46         0   114990          2     0      0     350      0.0       0.0  1              1.0
```
- Required actual columns: PASS

## 4. First temporal snapshot dry-run

- Story: `46`
- Time bin: `0`
- Absolute window start: `1246498976`
- Snapshot node universe: `247,929`
- Directed edges with `friend_date < window_start`: `1,622,850`
- Exposure rows in snapshot: `108`
- Positive samples in snapshot: `18`
- Latest included edge time: `1246498949`
- Included edges at/current-after window start: `0`
- Past edges incorrectly excluded by cutoff: `0`
- Strict past-only edge check: PASS

## 5. One-percent training smoke run

- Full train rows: `267,702`
- Smoke rows (ceil 1%, seed 42): `2,678`
- Smoke story-time groups: `1,262`
- Epochs: `1`
- Weighted BCE: `0.10407423743`
- Output probability shape: `(2678,)`
- Finite forward/backward loss and predictions: PASS

## Gate result

**PASS — all five Stage 1 checks completed. No full model training was run.**

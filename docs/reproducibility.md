# Reproducibility notes

All experiment entry points accept explicit random seeds and path arguments.
Paths default relative to the repository root; no local absolute path is
required.

The published showcase results used:

- Friendster: 2,368 nodes and 66,848 undirected edges;
- YouTube: 3,269 nodes and 14,510 undirected edges;
- Orkut: 4,621 nodes and 36,745 undirected edges;
- fixed `beta=5`;
- synchronous irreversible updates;
- seed nodes sampled from community 0.

The intervention confidence bands use 10 independent repeats with 50 cascades
per repeat at each intervention level. Results are deterministic given the same
input CSV files, software environment, and random seed.

Use the built-in demo only for installation and interface testing. It is not
part of the reported empirical-topology results.

## Recorded software and randomization

- Recommended interpreter: Python 3.10; code supports Python 3.9 or newer.
- Formal Digg community preparation: `igraph==1.0.0`, using built-in `Graph.community_leiden`; no separate `leidenalg` import.
- Synthetic default random seed: `12345`; fixed `beta=5`, five initial seeds, 20 maximum steps.
- Parameter grid: `a={0.6,0.8,1.0}`, `b={0.2,0.4,0.6}`, `theta={0.05,0.10,0.15}`, 100 cascades per setting.
- Digg cleaning/community seed: `20240821`.
- Digg pilot selection, negative sampling, story splits, XGBoost, grouped CV, and Platt partition seed: `42`.
- Digg exposure window: 3,600 seconds; maximum negative sampling ratio: 5:1; pilot size: 100 cascades.
- XGBoost: 400 trees, learning rate 0.05, depth 4, minimum child weight 10, row subsampling 0.8, feature subsampling 0.9, L2 penalty 1, histogram tree method.

## Digg data fingerprints

These hashes identify the local inputs and deterministic processed files used for the reported pilot. The files themselves are ignored by Git.

| file | SHA-256 |
|---|---|
| `data/raw/digg2009/digg_votes.csv` | `9e8a096c081c42ac199767fe5f436313f5dcce3198cbbeb4b735567276074bd0` |
| `data/raw/digg2009/digg_friends.csv` | `87688cfdb4904e120e32d5c68c45ee09acd5fbbdea1c1ff415ac6d3cedd085f2` |
| `data/processed/digg_votes_clean.csv.gz` | `65212ab6bfe83bac6e5d10caadbb92b5d1372cfdad7987094e6040f6d7040a25` |
| `data/processed/digg_friends_clean.csv.gz` | `32331951bd266f268e77de28d656f0090587be9bd53bd74a3b2f1316f09d8890` |
| `data/processed/digg_communities.csv` | `1c17e3714a0eedd0aaec2795eb60c4e6cac46c337dafde1fa4ee943c6d104325` |
| `data/processed/digg_exposure_pilot.csv.gz` | `058c32defc44251c96c7f61fb12d266889b17fa59f1464018a184baa714fee27` |

Key public results are `outputs/parameter_recovery.csv`, `outputs/parameter_grid_recovery.csv`, `outputs/unified_b_intervention.csv`, `outputs/digg/controlled_model_coefficients.csv`, `outputs/digg/xgb_group_cv_metrics.csv`, and `outputs/digg/final_model_metrics.csv`. The last two XGBoost CSV hashes are respectively `84158334c82c2568694c171af48e640053ccfe71e60e9cecc1c3805307ef567b` and `604b9b3bc2ba6f77ad95c59d993f4415f88c0ad93bc6a30a1120f6b8bc41de`.

## Execution levels

- `python scripts/run_all.py --demo --quick --outputs-dir outputs/smoke` is the fast, data-free smoke test.
- `python scripts/run_digg_pipeline.py` is the long pipeline and requires local raw Digg data.
- Compact precomputed summaries and figures are included for inspection without rebuilding row-level data.
- The final repository audit did not rerun the complete Digg pipeline.

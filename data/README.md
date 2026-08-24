# Data preparation

This repository uses SNAP network inputs for the synthetic experiments and the Digg 2009 archive for the observational cascade case study. Neither raw data family is redistributed here. Keep downloaded and generated row-level data local, and commit only this schema documentation and compact result artifacts.

## Synthetic SNAP network inputs

| network | original SNAP page | edge file | community file |
|---|---|---|---|
| YouTube | https://snap.stanford.edu/data/com-Youtube.html | `edges_youtube_remapped.csv` | `community_youtube_remapped.csv` |
| Friendster | https://snap.stanford.edu/data/com-Friendster.html | `edges_friendster_remapped.csv` | `community_friendster_remapped.csv` |
| Orkut | https://snap.stanford.edu/data/com-Orkut.html | `edges_orkut_remapped.csv` | `community_orkut_remapped.csv` |

Download each network from SNAP and follow its terms of use and citation instructions. The `community_*_remapped.csv` inputs are derived two-community labels and are not upstream SNAP files.

Place the prepared files directly in `data/`. Edge files use `source,target`; community files use `id,community`, with labels `0` and `1`. Node IDs must be integers. Each undirected edge appears once, and every edge-list node must have exactly one community label. The scripts validate schemas, self-loops, duplicate edges, duplicate labels, and missing labels.

A reproducible conversion from overlapping SNAP groups should record how the two groups were selected without simulation outcomes, how overlapping nodes were assigned, whether other nodes were removed, and whether the graph was induced or sampled.

For a node-community mapping that already gives one `node_id,community_id` pair per row, generate a deterministic induced subgraph with:

```bash
python scripts/prepare_snap_two_community.py \
  --edge-list /path/to/raw_edges.txt \
  --community-mapping /path/to/node_community.txt \
  --community-a COMMUNITY_ID_A \
  --community-b COMMUNITY_ID_B \
  --network friendster \
  --output-dir data
```

The script accepts comma- or whitespace-delimited inputs, removes self-edges and duplicate undirected edges, sorts nodes and edges, maps the two requested communities to `0/1`, and writes the two expected CSV files plus a SHA-256 manifest. A node found in both selected communities is rejected as ambiguous rather than assigned arbitrarily. Current formal-input hashes and the remaining historical provenance limitation are recorded in [the SNAP provenance audit](../docs/snap_input_provenance.md).

## Digg 2009 real-cascade data

Download the archive from the [official Figshare dataset page](https://figshare.com/articles/dataset/Digg_2009_social_news_votes_and_graph/2062467) (DOI: [10.6084/m9.figshare.2062467](https://doi.org/10.6084/m9.figshare.2062467)). The archive's public password is `digg2009_user`. The Figshare record is licensed CC BY 4.0 and its description also says that the data are made available for research purposes; users should follow the current terms displayed on the official record.

Place the extracted, headerless files here:

```text
data/raw/digg2009/
├── digg_votes.csv
└── digg_friends.csv
```

Expected source layouts are:

| file | fields | meaning |
|---|---|---|
| `digg_votes.csv` | `vote_date,voter_id,story_id` | Unix vote time, voter identifier, story identifier |
| `digg_friends.csv` | `mutual,friend_date,user_id,friend_id` | mutual-link indicator, Unix link time, follower, followed user |

The directed relation `user_id -> friend_id` means that `user_id` follows or watches `friend_id`. The analysis therefore reverses it for information influence: `friend_id -> user_id`. A `friend_date` of zero is treated as unknown and excluded from the pre-vote baseline network.

The official record reports 3,018,197 votes on 3,553 stories by 139,409 users and 1,731,658 friendship links involving 71,367 distinct users. Cite both the dataset record and the source paper:

> Hogg, T. and Lerman, K. (2012). “Social Dynamics of Digg.” *EPJ Data Science*, 1(5). https://doi.org/10.1140/epjds5

## Files intentionally excluded from Git

The repository does not upload:

- `data/raw/`, including the downloaded Digg archive;
- `data/processed/`, including cleaned votes, network, communities, and pilot exposure data;
- full or row-level exposure tables (`outputs/exposure_table*`, `*.csv.gz`, or Parquet files); or
- serialized fitted objects (`*.pkl` and `*.joblib`).

These files can be regenerated locally with the commands documented in the project README. Compact aggregate CSVs, Markdown reports, and final PNG figures remain eligible for version control.

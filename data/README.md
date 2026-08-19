# Data preparation

Large SNAP files and derived node-level data are intentionally not included in
this repository. Download source networks from the Stanford Network Analysis
Project:

- [Friendster](https://snap.stanford.edu/data/com-Friendster.html)
- [YouTube](https://snap.stanford.edu/data/com-Youtube.html)
- [Orkut](https://snap.stanford.edu/data/com-Orkut.html)

Please follow SNAP's terms, citation guidance, and the terms of the underlying
datasets. This repository does not relicense or redistribute them.

## Expected files

For each network name `<network>`, place two consistently remapped CSV files in
this directory:

```text
edges_<network>_remapped.csv
community_<network>_remapped.csv
```

The default network names are `friendster`, `youtube`, and `orkut`.

### Edge schema

```csv
source,target
1,17
1,42
```

- `source` and `target` are integer node IDs.
- Each undirected edge appears exactly once.
- Self-loops and duplicate undirected edges are not allowed.

### Community schema

```csv
id,community
1,0
2,0
3,1
```

- Every edge-list node has exactly one integer community label.
- The node-ID set must match the edge list.
- The reported experiments use two-community subgraphs with labels `0` and `1`.
- Overlapping source communities must be resolved consistently before running
  the experiments.

The scripts validate schemas, duplicate IDs, edge duplication, and missing
community labels before simulation. They use the supplied community labels and
do not run Louvain as a fallback.

## Why data are excluded

Raw and derived graph files can be large and may carry dataset-specific usage
conditions. Keeping them outside Git avoids unnecessary redistribution,
repository bloat, and accidental publication of local or sensitive data. Only
aggregate, synthetic-experiment summaries are committed in `outputs/`.

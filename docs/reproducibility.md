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

# Recommendation training summary

- Seed: `42`
- Eligible users: `64,939`
- Training interactions: `2,744,579`
- Validation/test candidate count: `64,930` / `64,931`
- BPR selection: `{"dimension": 64, "lr": 0.001, "max_epochs": 12, "regularization": 1e-05, "selected_epoch": 11}`
- DeepFM selection: `{"dropout": 0.2, "embedding_dim": 16, "hidden": [64, 32], "lr": 0.001, "max_epochs": 12, "weight_decay": 1e-05}`
- Final v3.1 recommendation model: `DeepFM_context`
- `training_seconds` for BPR-MF and DeepFM_full includes validation search; ID/context rows are fixed-configuration fit times. Inference time covers the complete warm-start test candidate set.
- Test was evaluated only after validation-based selection.
- Candidate and input checksums: see `candidate_manifest.json`.

The experiment is an implicit-feedback, sampled-candidate offline evaluation. It is not CTR prediction and sampled non-interactions are not explicit negative feedback.

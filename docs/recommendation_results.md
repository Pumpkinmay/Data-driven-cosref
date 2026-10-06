# Digg v3.1 recommendation results

## Warm-start sampled-candidate results

| Model | NDCG@10 | Recall@10 | MRR | AUC |
|---|---:|---:|---:|---:|
| Popularity | 0.108374 | 0.215860 | 0.097903 | 0.526530 |
| BPR-MF | 0.662113 | 0.905530 | 0.589556 | 0.960892 |
| DeepFM_ID | 0.534932 | 0.790023 | 0.466476 | 0.932601 |
| **DeepFM_context** | **0.727087** | **0.928401** | **0.665968** | **0.970299** |
| DeepFM_full | 0.720863 | 0.927846 | 0.658026 | 0.969797 |

Additional calculated metrics and measured model size/runtime are:

| Model | NDCG@20 | Recall@20 | HitRate@10 | Parameters | Training seconds | Inference seconds |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 0.14353039 | 0.3561781 | 0.21585991 | 0 | 0 | 5.589613 |
| BPR-MF | 0.67522711 | 0.95660008 | 0.90553049 | 4383488 | 122.66316 | 0.647883 |
| DeepFM_ID | 0.56452165 | 0.90585391 | 0.79002326 | 1168590 | 81.595159 | 1.1561327 |
| **DeepFM_context** | **0.737165** | **0.96778118** | **0.92840092** | 1168850 | 177.02991 | 11.363019 |
| DeepFM_full | 0.73103643 | 0.96759637 | 0.92784648 | 1233386 | 591.04363 | 8.7000615 |

All means use one held-out target per evaluated user. User-level bootstrap confidence intervals are in `outputs/recommendation/bootstrap_intervals.csv`.

BPR-MF improves NDCG@10 over Popularity by 0.553739. DeepFM_full changes NDCG@10 relative to BPR-MF by 0.058750. Paired user-bootstrap intervals for both model deltas are included in the same CSV.

## Coverage

- Validation warm-user: 100.0000%; warm-item: 100.0000%; cold-item: 0.0000%.
- Test warm-user: 100.0000%; warm-item: 100.0000%; cold-item: 0.0000%.

Cold target items are reported as coverage, not silently scored with trained ID embeddings.

## DeepFM ablation

| model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | HitRate@10 | MRR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepFM_ID | 0.79002326 | 0.90585391 | 0.53493182 | 0.56452165 | 0.79002326 | 0.46647582 | 0.93260149 |
| DeepFM_context | 0.92840092 | 0.96778118 | 0.72708665 | 0.737165 | 0.92840092 | 0.66596807 | 0.97029901 |
| DeepFM_full | 0.92784648 | 0.96759637 | 0.72086344 | 0.73103643 | 0.92784648 | 0.65802615 | 0.969797 |

`DeepFM_ID` uses user/story IDs; `DeepFM_context` adds activity, popularity, and time; `DeepFM_full` additionally adds the pre-period baseline user community. Context raises NDCG@10 over the ID-only variant by 0.192155. Adding community changes NDCG@10 relative to context by -0.006223; it does not provide an additional gain in this run. This is a predictive comparison, not a causal community effect. ID embeddings are identifiers, not content semantics.

DeepFM_context is therefore the final v3.1 recommendation model. The lack of an incremental recommendation gain from the community field does not negate the separate mechanistic value of community structure in diffusion analysis. DIN is not implemented.

## Leakage audit

All eight checks in the protocol pass: disjoint leave-last-two-out rows; strict-past user activity and training-story popularity; no future-appearing test negatives; one shared candidate set across models; validation-only selection; test used after selection; and user-level bootstrap resampling. See [the protocol](recommendation_protocol.md) for implementation evidence and the per-user/global-time caveat.

## Limitations

This is sampled-candidate offline evaluation and not full-catalog online ranking. It is not CTR prediction. Digg provides votes but no impression or click-opportunity log, so sampled non-interactions are not confirmed negatives and the results do not imply online CTR improvement. No DIN model is included and no unavailable story text/category features are fabricated. These recommendation metrics must not be compared numerically with v2.0 diffusion PR-AUC because the targets and candidate universes differ.

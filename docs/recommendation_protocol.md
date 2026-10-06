# Digg v3.1 recommendation protocol

## Task

This is an implicit-feedback next-story ranking task. Given a user's votes strictly before time `t`, the models rank stories for the user's next vote. Digg has no impression log; unvoted candidates are **sampled non-interactions**, not observed negative feedback, and this is not CTR prediction.

## Split and candidate construction

- Users have at least five cleaned votes.
- Per user, events are stably ordered by `(timestamp, story_id, original input order)`. The last event is test, the penultimate event is validation, and earlier events are training.
- User and story mappings are determined from training interactions only.
- Each warm-start validation/test positive shares the same 99 negatives across all models. A negative story must have first appeared in the complete cleaned vote table strictly before the target timestamp and must not have been voted by that user strictly before the target.
- Very early warm targets with fewer than 99 distinct legal negatives are excluded rather than padded with duplicates or future stories. Validation excludes `9` of `64939` warm targets; test excludes `8` of `64939`.
- Seed: `42`. Validation candidate SHA-256: `a1c6b6c8dd0c2ab35b82c8853e054d6fc2d5dec746649385e19b3a34d89078d8`. Test candidate SHA-256: `6084b9c78ae22f9af02afefd337aaed2564be39c7469a5ab375ed988b50dd85b`.
- Primary metrics cover warm-start target items only. Cold-item rates are reported separately because ID-based BPR-MF and DeepFM cannot infer unseen item embeddings.

## Models and tuning

Popularity counts only training interactions strictly before each target timestamp. BPR-MF uses temporal pairwise negatives and BPR loss. BPR embedding dimension, learning rate, regularization, and stopping epoch are selected using validation NDCG@10. DeepFM has linear, FM, and DNN components. Its full variant is tuned using validation NDCG@10; the selected configuration is then fixed for the ID/context/community ablation. Test is evaluated once after selection.

DeepFM numeric features are `log1p(user activity before t)`, `log1p(training-story popularity before t)`, UTC hour, and relative day. Its sparse fields are training-mapped user/story IDs and, for `DeepFM_full`, the baseline community label. The numeric scaler is fitted on training examples only. ID embeddings are identifiers, not content semantics.

Community labels come from the baseline friendship network constructed before the voting period, so they are available before the evaluated interactions. The requested leave-last-two-out split is per user rather than a single global calendar cutoff; candidate eligibility and dynamic statistics are still computed strictly before each target timestamp.

## Final leakage audit

1. **Pass — split isolation:** mutually exclusive position masks put earlier interactions in train, the penultimate interaction in validation, and the last interaction in test; validation/test positives are not train rows.
2. **Pass — user activity:** both training and scoring use strict-past counts; equal-time events are excluded by `rank(method="min") - 1` or left-sided timestamp search.
3. **Pass — story popularity:** popularity is counted only from training interactions with timestamps strictly below the target timestamp.
4. **Pass — temporal candidates:** candidate validation checks that every sampled story first appeared in the complete cleaned vote table strictly before the target; eight test targets with fewer than 99 legal negatives are excluded rather than padded.
5. **Pass — shared candidates:** Popularity, BPR-MF, and every DeepFM ablation score the same in-memory validation and test `CandidateSet` objects.
6. **Pass — validation-only selection:** BPR and DeepFM configuration/stopping choices use validation NDCG@10 only.
7. **Pass — final test use:** test scoring occurs only after configuration selection in the final pipeline. Test metrics are not consulted by the selection functions.
8. **Pass — user bootstrap:** leave-last-two-out contributes one test target per evaluated user, and bootstrap rows resample those user-level metric vectors.

The audit verifies the implemented per-user protocol; it does not turn leave-last-two-out into a global calendar split.

## Interpretation limits

This is sampled-candidate offline evaluation, not full-catalog online ranking. There are no true exposure logs; sampled negatives are not explicit negatives. Results do not establish online CTR lift and are not directly comparable to v2.0 diffusion PR-AUC. DIN is not implemented, and no missing story text or category features are invented.

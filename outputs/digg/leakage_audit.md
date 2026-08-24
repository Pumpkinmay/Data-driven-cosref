# Digg XGBoost leakage audit

## Result

**PASS — no train/test, temporal-exposure, control-variable, tuning, scaling, or calibration leakage was detected.**

## Checks

1. **Story-level separation:** 80 train and 20 test stories; story-ID overlap **0**; rows **267,702 / 78,636**. Every row inherits its story's saved split.
2. **User history:** all 346,338 rows use `bisect_left(user_vote_times, window_start)`, so only timestamps strictly earlier than the absolute window start are counted; current-story votes are excluded from the other-story activity count. Negative-count/time-boundary violations: **0**.
3. **Cascade size:** all rows use `bisect_left(network_covered_vote_times, window_start)`; time-zero rows with a nonzero prior cascade size: **0**.
4. **Network exposure:** independently reconstructed `m_in/m_out` from baseline `friend_id -> user_id` edges and adopters with `vote_date < window_start`. Count mismatches: **0**; degree mismatches: **0**; detected current/future inclusions: **0**.
5. **Preprocessing/tuning/calibration:** XGBoost uses raw features and **no scaler**. Hyperparameters are fixed in code; there is no search, test-driven early stopping, or probability calibration in ablation/CV fits. Platt calibration uses **64 model-fit + 16 calibration stories**, all drawn only from the saved 80 training stories; overlap with the 20 held-out stories is **0**.

## Baseline-network boundary

Friendships are restricted to `0 < friend_date <= 1243770221`, the earliest vote timestamp. Thus exposure construction does not import friendships formed after cascades begin.

The audit verifies implementation-time separation and temporal ordering. It cannot eliminate unobserved confounding or make observational predictors causal.

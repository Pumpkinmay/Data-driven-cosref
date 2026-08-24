# Digg XGBoost five-fold story-group cross-validation

## Design and integrity

- Models are limited to the existing `XGB_base`, `XGB_context`, and `XGB_full`; no model or feature was added or reselected.
- Five-fold `GroupKFold`, shuffled with seed 42; `group=story_id`. Each fold contains 80 train and 20 validation stories, and every story is validation data exactly once.
- Story overlap is zero in all five folds. All fits and metrics use the original `sampling_weight`.
- XGBoost hyperparameters are unchanged and fixed before validation; no fold is used for tuning or feature selection.
- Cluster bootstrap: **1,000** replicates, resampling the 100 out-of-fold validation stories with replacement and recomputing pooled weighted PR-AUC for context and full.

## Per-fold results

The sampling-weighted positive prevalence is the expected PR-AUC of a random ranking under the reconstructed target population.

| fold | model | weighted prevalence / random PR baseline | log loss | Brier | ROC-AUC | PR-AUC | full−context PR-AUC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | XGB_base | 0.0001792684114 | 0.001641418133 | 0.0001792007393 | 0.774683524 | 0.0006229798044 | — |
| 1 | XGB_context | 0.0001792684114 | 0.001513180864 | 0.0001811447014 | 0.9074012289 | 0.007216647273 | — |
| 1 | XGB_full | 0.0001792684114 | 0.001492765359 | 0.0001796144555 | 0.9094185323 | 0.01978027544 | 0.01256362817 |
| 2 | XGB_base | 0.0001166309506 | 0.001119849255 | 0.0001165936165 | 0.764308077 | 0.0005972068739 | — |
| 2 | XGB_context | 0.0001166309506 | 0.0009535754815 | 0.0001163197401 | 0.90734783 | 0.01066138275 | — |
| 2 | XGB_full | 0.0001166309506 | 0.0009633335453 | 0.0001189065617 | 0.9098856099 | 0.01571795945 | 0.0050565767 |
| 3 | XGB_base | 0.0001158210778 | 0.001119096672 | 0.000115791456 | 0.7553000106 | 0.0004764354792 | — |
| 3 | XGB_context | 0.0001158210778 | 0.0009653057622 | 0.0001165925873 | 0.9168099411 | 0.01244467261 | — |
| 3 | XGB_full | 0.0001158210778 | 0.0009639511275 | 0.0001184758649 | 0.9190429069 | 0.007984924412 | -0.0044597482 |
| 4 | XGB_base | 0.0001691028969 | 0.001568369186 | 0.0001690409785 | 0.7728083133 | 0.0006303028874 | — |
| 4 | XGB_context | 0.0001691028969 | 0.001391668737 | 0.0001683289679 | 0.8957126841 | 0.01109981042 | — |
| 4 | XGB_full | 0.0001691028969 | 0.0013909505 | 0.0001690183022 | 0.8970749292 | 0.01153169731 | 0.0004318868921 |
| 5 | XGB_base | 0.0001057206899 | 0.001036896865 | 0.0001056949487 | 0.7344300856 | 0.000483374393 | — |
| 5 | XGB_context | 0.0001057206899 | 0.0009079742362 | 0.0001057563032 | 0.8808384909 | 0.01529453343 | — |
| 5 | XGB_full | 0.0001057206899 | 0.0009329547299 | 0.0001108207917 | 0.8831282153 | 0.01004465936 | -0.005249874071 |

## Mean ± standard deviation across folds

| model | prevalence | log loss | Brier | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| XGB_base | 0.0001373088053 ± 3.412688803e-05 | 0.001297126022 ± 0.0002841437258 | 0.0001372643478 ± 3.41079403e-05 | 0.7603060021 ± 0.01638337974 | 0.0005620598876 ± 7.603716247e-05 |
| XGB_context | 0.0001373088053 ± 3.412688803e-05 | 0.001146341016 ± 0.0002835084658 | 0.00013762846 ± 3.445500077e-05 | 0.901622035 ± 0.01381785472 | 0.0113434093 ± 0.00293167308 |
| XGB_full | 0.0001373088053 ± 3.412688803e-05 | 0.001148791052 ± 0.0002702337526 | 0.0001393671952 ± 3.228394519e-05 | 0.9037100387 ± 0.01390555248 | 0.01301190319 ± 0.00472756979 |

## Exposure increment and bootstrap

- Per-fold XGB_full minus XGB_context PR-AUC deltas: **+0.012563628, +0.0050565767, -0.0044597482, +0.00043188689, -0.0052498741**.
- XGB_full has higher PR-AUC in **3/5 folds**. Therefore the requested **at least 4/5 folds** criterion is **not met**.
- Mean fold delta: **+0.001668493898**; fold SD: **0.007367689924**.
- Story-cluster bootstrap pooled PR-AUC delta: median **+0.001479264594**, percentile 95% interval **[-0.003924021637, +0.005472954944]**.
- Bootstrap probability that the increment is positive: **83.10%**.

The exposure features improve mean PR-AUC, but the fold-win criterion and cluster interval determine whether that gain is stable across cascades. These are predictive, observational results and do not give `m_in` or `m_out` a causal interpretation.

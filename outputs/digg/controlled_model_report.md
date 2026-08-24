# Digg controlled-model diagnostic report

## Leakage-safe controls

- `user_prior_votes`: votes by the same node in other stories with timestamp strictly earlier than the current window's absolute start.
- `cascade_size_so_far`: community-labelled adopters in the current story strictly before the current window start.
- Transformations: `log_user_activity=log1p(user_prior_votes)` and `log_cascade_size=log1p(cascade_size_so_far)`.
- User-control leakage violations: **0**
- Cascade-control leakage violations: **0**
- Rows: **346,338**; train rows: **267,702**; test rows: **78,636**
- The existing 80/20 story split and original sampling weights were reused.
- No `frac_in`/`frac_out`, coefficient constraints, class weights, or interventions were used.

## Full-population coefficients

All coefficients below are converted back to their original feature scales.

| model | time control | coef_m_in | coef_m_out | coef_degree | log_user_activity | log_cascade_size | converged/iterations |
|---|---|---:|---:|---:|---:|---:|---|
| D0 | log1p_time_bin | NA | NA | 0.002369615217 | NA | NA | yes/2 |
| D1 | log1p_time_bin | 0.02710798787 | -0.04863491722 | 0.00198628597 | NA | NA | yes/5 |
| D2 | log1p_time_bin | 0.008668992925 | 0.01496748397 | 0.0002356261747 | 0.8847156869 | NA | yes/7 |
| D3 | log1p_time_bin | 0.005945365101 | -0.01014412869 | 0.0005294109099 | 0.8788255875 | 0.639104607 | yes/6 |
| D3 | piecewise_time | 0.004289373752 | -0.01310886502 | 0.0006306811127 | 0.8784287049 | 0.5004317505 | yes/11 |

## Time-control sensitivity

The piecewise D3 uses 0–1h as the reference and indicators for 1–3h, 3–6h, 6–12h, 12–24h, and 24h+.

| D3 time control | coef_m_in | coef_m_out | coef_degree |
|---|---:|---:|---:|
| log1p(time_bin) | 0.005945365101 | -0.01014412869 | 0.0005294109099 |
| piecewise categories | 0.004289373752 | -0.01310886502 | 0.0006306811127 |

## Held-out metrics

PR-AUC is sampling-weighted average precision. Full models are evaluated on all, zero-exposure, and exposed test rows; exposed-only fits are evaluated only on exposed rows.

| fit scope | model | time control | test subset | rows | weighted y rate | log loss | Brier | ROC-AUC | PR-AUC |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| full | D0 | log1p_time_bin | all_test | 78,636 | 0.01516731% | 0.001457248637 | 0.0001516466241 | 0.7062303744 | 0.0003183706256 |
| full | D0 | log1p_time_bin | zero_exposure_test | 67,025 | 0.01233733% | 0.001220158262 | 0.0001233624148 | 0.6825593945 | 0.0002467071839 |
| full | D0 | log1p_time_bin | exposed_test | 11,611 | 0.03632647% | 0.00322992153 | 0.0003631214724 | 0.7414228115 | 0.00111553236 |
| full | D1 | log1p_time_bin | all_test | 78,636 | 0.01516731% | 0.001454208076 | 0.0001516476798 | 0.7090371881 | 0.0003453067552 |
| full | D1 | log1p_time_bin | zero_exposure_test | 67,025 | 0.01233733% | 0.001220357586 | 0.0001233614273 | 0.6824826248 | 0.0002474387917 |
| full | D1 | log1p_time_bin | exposed_test | 11,611 | 0.03632647% | 0.003202657052 | 0.0003631378049 | 0.7628449078 | 0.001092028684 |
| full | D2 | log1p_time_bin | all_test | 78,636 | 0.01516731% | 0.001318031077 | 0.0001514483447 | 0.8486424549 | 0.003759386259 |
| full | D2 | log1p_time_bin | zero_exposure_test | 67,025 | 0.01233733% | 0.001119287774 | 0.0001232645552 | 0.8326801435 | 0.003208515374 |
| full | D2 | log1p_time_bin | exposed_test | 11,611 | 0.03632647% | 0.002803991295 | 0.0003621723762 | 0.8746327793 | 0.005721186674 |
| full | D3 | log1p_time_bin | all_test | 78,636 | 0.01516731% | 0.001261366429 | 0.0001513730246 | 0.8918300001 | 0.0038509704 |
| full | D3 | log1p_time_bin | zero_exposure_test | 67,025 | 0.01233733% | 0.00106090995 | 0.0001231778482 | 0.8853308428 | 0.003876779647 |
| full | D3 | log1p_time_bin | exposed_test | 11,611 | 0.03632647% | 0.002760135689 | 0.000362182194 | 0.8914079046 | 0.006483274301 |
| full | D3 | piecewise_time | all_test | 78,636 | 0.01516731% | 0.001297337216 | 0.0001513676217 | 0.8584292957 | 0.004212989824 |
| full | D3 | piecewise_time | zero_exposure_test | 67,025 | 0.01233733% | 0.001091705893 | 0.0001231861266 | 0.8488948543 | 0.003601161949 |
| full | D3 | piecewise_time | exposed_test | 11,611 | 0.03632647% | 0.002834797655 | 0.0003620744985 | 0.8619704474 | 0.007254743223 |
| exposed_only | D1 | log1p_time_bin | exposed_test | 11,611 | 0.03632647% | 0.003043698626 | 0.0003629017504 | 0.7844102521 | 0.001987542834 |
| exposed_only | D2 | log1p_time_bin | exposed_test | 11,611 | 0.03632647% | 0.00276248281 | 0.0003630755859 | 0.8849350026 | 0.006553165395 |
| exposed_only | D3 | log1p_time_bin | exposed_test | 11,611 | 0.03632647% | 0.002734085294 | 0.0003627241459 | 0.8925285464 | 0.007008668911 |

## Exposed-only association fits

These coefficients condition on `m_in+m_out>0`; they are exposed-only associations and are not models for the full user population.

| model | coef_m_in | coef_m_out | coef_degree | log_user_activity | log_cascade_size |
|---|---:|---:|---:|---:|---:|
| D1 | 0.03631399049 | -0.04430993665 | -0.0007388730112 | NA | NA |
| D2 | 0.01156495264 | 0.02759555146 | -0.0009865912658 | 0.7894729727 | NA |
| D3 | 0.01074926467 | 0.01979542519 | -0.0008314328209 | 0.7924723548 | 0.23478668 |

## Story-cluster bootstrap

- Requested fits: **600** (200 replicates × D1/D2/D3)
- Converged fits: **600**
- Each replicate resamples the same 80 training stories with replacement and uses cluster multiplicity times the original sampling weight.
- Intervals are percentile 95% intervals over converged fits. Positive/negative shares are computed without imposing signs.

| model | quantity | point | 2.5% | median | 97.5% | positive | negative | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | coef_m_in | 0.02710798787 | 0.02169227614 | 0.02765730942 | 0.0355343944 | 100.00% | 0.00% | 200 |
| D1 | coef_m_out | -0.04863491722 | -0.0706226004 | -0.04723589667 | -0.02540828574 | 0.00% | 100.00% | 200 |
| D1 | coef_degree | 0.00198628597 | 0.001595674544 | 0.001939748146 | 0.002393529813 | 100.00% | 0.00% | 200 |
| D1 | a_eff | 0.005421597574 | 0.004338455227 | 0.005531461885 | 0.00710687888 | 100.00% | 0.00% | 200 |
| D1 | b_eff | -0.009726983444 | -0.01412452008 | -0.009447179334 | -0.005081657147 | 0.00% | 100.00% | 200 |
| D1 | theta_eff | -0.000397257194 | -0.0004787059625 | -0.0003879496291 | -0.0003191349088 | 0.00% | 100.00% | 200 |
| D2 | coef_m_in | 0.008668992925 | 0.001096246475 | 0.00887275288 | 0.01654518755 | 99.00% | 1.00% | 200 |
| D2 | coef_m_out | 0.01496748397 | -0.006851910875 | 0.01297181116 | 0.03794822741 | 88.50% | 11.50% | 200 |
| D2 | coef_degree | 0.0002356261747 | -0.0002595097059 | 0.0002244053556 | 0.0007457763561 | 80.50% | 19.50% | 200 |
| D2 | a_eff | 0.001733798585 | 0.0002192492949 | 0.001774550576 | 0.00330903751 | 99.00% | 1.00% | 200 |
| D2 | b_eff | 0.002993496793 | -0.001370382175 | 0.002594362232 | 0.007589645483 | 88.50% | 11.50% | 200 |
| D2 | theta_eff | -4.712523493e-05 | -0.0001491552712 | -4.488107111e-05 | 5.190194117e-05 | 19.50% | 80.50% | 200 |
| D3 | coef_m_in | 0.005945365101 | -0.002376695426 | 0.005743643668 | 0.01356955071 | 91.00% | 9.00% | 200 |
| D3 | coef_m_out | -0.01014412869 | -0.03375412384 | -0.01149971113 | 0.01196782547 | 19.50% | 80.50% | 200 |
| D3 | coef_degree | 0.0005294109099 | 6.140195396e-05 | 0.0005201557001 | 0.001014358709 | 98.50% | 1.50% | 200 |
| D3 | a_eff | 0.00118907302 | -0.0004753390851 | 0.001148728734 | 0.002713910143 | 91.00% | 9.00% | 200 |
| D3 | b_eff | -0.002028825738 | -0.006750824767 | -0.002299942227 | 0.002393565094 | 19.50% | 80.50% | 200 |
| D3 | theta_eff | -0.000105882182 | -0.0002028717419 | -0.00010403114 | -1.228039079e-05 | 1.50% | 98.50% | 200 |

## Answers

1. **After controlling user history, is degree still positive?** The D2 point estimate is positive (`0.000235626`), but its 95% interval `[-0.00025951, 0.000745776]` crosses zero and only 80.50% of replicates are positive. In D3, degree is more stable: 98.50% positive.
2. **After flexible time and cascade-heat controls, is m_out still negative?** Both D3 point estimates are negative: log-time `-0.0101441` and piecewise time `-0.0131089`. This is not bootstrap-stable evidence of a negative effect: the log-time D3 interval `[-0.0337541, 0.0119678]` crosses zero and 80.50% of replicates are negative.
3. **Are exposure signs bootstrap-stable?** Only in the uncontrolled D1: `m_in` is positive in 100.0% and `m_out` negative in 100.0%. After controls, `m_in` is positive in D2/D3 at 99.0%/91.0%; `m_out` is negative at only 11.5%/80.5%. D2's `m_out` point estimate reverses positive, and both D2/D3 `m_out` intervals cross zero. No theoretical sign was imposed.
4. **Is M1 clearly better for exposed users?** It has more signal there, but not a uniformly decisive gain: full-fit D1 improves exposed-test log loss by 2.73e-05 and ROC-AUC by 0.02142, while Brier/PR changes are mixed. In exposed-only D2/D3, `m_out` becomes positive and degree becomes negative, further showing sensitivity to conditioning and controls. Exposed-only fits are associations conditional on exposure.
5. **Can Digg estimate effective paper a, b, theta?** Not with a defensible structural interpretation in this pilot. The signs of `b_eff` and `theta_eff` remain contrary to the paper-parameter interpretation after controls. Digg currently supports predictive and associational validation only, not recovery of causal or structural a/b/theta.

This remains a 100-cascade pilot. No fit was expanded to all 3,553 cascades, no coefficient was clipped, and no real-data intervention was run.

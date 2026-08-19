# Method note

## Synchronous normalized-threshold contagion

Let `G=(V,E)` be an undirected network with observed community label `c_i` for
each node. At time `t`, inactive node `i` has

```text
m_in,i(t)  = sum_{j in N(i)} x_j(t) 1[c_j = c_i]
m_out,i(t) = sum_{j in N(i)} x_j(t) 1[c_j != c_i].
```

Its activation probability is

```text
p_i(t) = sigmoid(beta * (a*m_in,i(t) + b*m_out,i(t) - theta*d_i)).
```

All outcomes at time `t` are sampled from the same pre-update state and applied
simultaneously. Activation is irreversible. A cascade stops when no new node
activates, all nodes are active, or `max_steps` is reached.

## Exposure likelihood and recovery

Each inactive node contributes one row with

```text
cascade_id, time, node, m_in, m_out, degree, y
```

where `y=1` denotes activation in the next synchronous update. The model implies

```text
logit P(y=1) = beta*a*m_in + beta*b*m_out - beta*theta*degree.
```

After fitting `y ~ m_in + m_out + degree`, fixed `beta` gives

```text
a_hat     =  coef(m_in) / beta
b_hat     =  coef(m_out) / beta
theta_hat = -coef(degree) / beta.
```

The fitted intercept is retained as a diagnostic; its population value is zero
under the generating model.

## Identifiability

Recovery requires both outcome classes and meaningful independent variation in
all three predictors. A large network or a large number of exposure rows is not
sufficient when almost every `m_out` value is zero. The Friendster high-threshold
failure illustrates this support problem: the effective information for `b`
comes from the small subset of activation-risk rows with cross-community active
neighbors.

## Intervention estimands

The intervention experiment scales `b` relative to `b_baseline`. It reports:

- mean final cascade size as a fraction of network nodes;
- global-cascade probability, where a global cascade reaches at least 50% of
  nodes;
- 95% normal-approximation confidence intervals across independent repeat-level
  estimates.

These are model-based simulation estimands, not causal effects estimated from
observed platform behavior.

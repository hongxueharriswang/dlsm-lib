# DLSM User Guide

**Complete documentation for the Dynamic Latent-State Model library**

Version 0.1.0 · Repository: [github.com/hongxueharriswang/dlsm-lib](https://github.com/hongxueharriswang/dlsm-lib)

---

## Table of Contents

**Part I — Getting Started**
1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Your First Five Minutes](#3-your-first-five-minutes)

**Part II — Core Concepts**
4. [The Latent State](#4-the-latent-state)
5. [Parameters and What They Mean](#5-parameters-and-what-they-mean)
6. [Observations and Measurement Channels](#6-observations-and-measurement-channels)

**Part III — Workflows**
7. [Simulating Data](#7-simulating-data)
8. [Filtering: Tracking Latent States](#8-filtering-tracking-latent-states)
9. [Estimating Parameters with PMCMC](#9-estimating-parameters-with-pmcmc)
10. [Working with Real Data](#10-working-with-real-data)

**Part IV — Advanced Topics**
11. [Custom Observation Channels](#11-custom-observation-channels)
12. [Custom Drift Functions](#12-custom-drift-functions)
13. [Diagnostics and Model Checking](#13-diagnostics-and-model-checking)
14. [Performance and Scaling](#14-performance-and-scaling)
15. [Reproducibility](#15-reproducibility)

**Part V — Reference**
16. [API Cheat Sheet](#16-api-cheat-sheet)
17. [Troubleshooting and FAQ](#17-troubleshooting-and-faq)
18. [Best Practices](#18-best-practices)

---

# Part I — Getting Started

## 1. Introduction

### 1.1 What problem does DLSM solve?

Traditional learning analytics treats engagement as a static quantity: you take a window of LMS activity, compute some counts, and predict an outcome. This works when learning is cohort-based and regular. It fails in **self-paced learning**, where:

- Students register at any time.
- LMS activity is irregular and sparse.
- Much learning happens offline (textbook reading, word processing).
- A student can pause for weeks and still return.

The Dynamic Latent-State Model (DLSM) instead treats engagement as a **latent, dynamic, multi-dimensional state** that evolves continuously and generates observations through multiple noisy channels. The library implements this model end-to-end: simulation, filtering, parameter estimation, and evaluation.

### 1.2 Who is this guide for?

- **Learning analytics researchers** who want to model engagement as a dynamic state.
- **Educational data scientists** who need Bayesian state-space inference on LMS data.
- **Developers** building engagement-aware interventions (early warning, adaptive pacing).

You should be comfortable with Python, NumPy, and basic probability. Familiarity with state-space models helps but is not required — Section 4 provides the conceptual grounding.

### 1.3 What you can do with the library

| Task | Entry point |
|------|-------------|
| Simulate synthetic student trajectories | `Simulator` |
| Track a student's engagement in real time | `ParticleFilter` |
| Estimate model parameters from data | `PMCMC` |
| Compare models / evaluate predictions | `metrics` |
| Visualize latent trajectories and posteriors | `plots` |
| Extend with custom observation channels | `ObservationModel` subclass |

---

## 2. Installation

### 2.1 From PyPI

```bash
pip install dlsm
```

### 2.2 From source

```bash
git clone https://github.com/hongxueharriswang/dlsm-lib.git
cd dlsm-lib
pip install -e .
```

### 2.3 Optional extras

The core library needs only NumPy and SciPy. Visualization and progress bars are optional:

```bash
pip install "dlsm[viz]"        # matplotlib
pip install "dlsm[progress]"   # tqdm
pip install "dlsm[all]"        # everything
```

### 2.4 Verifying the installation

```python
import dlsm
print(dlsm.__version__)          # 0.1.0

from dlsm import DLSM, DLSParams
model = DLSM(DLSParams.default())
print(model.params.alpha_O)      # 0.3
```

If that runs without error, you are ready.

### 2.5 Development environment

For contributing or reproducing the paper's simulation study:

```bash
pip install -r requirements-dev.txt
pip install -r requirements-lock.txt   # exact pinned versions
pytest tests/ -v
```

---

## 3. Your First Five Minutes

A complete end-to-end run in twenty lines:

```python
import numpy as np
from dlsm import DLSM, DLSParams, ParticleFilter, Simulator

# 1. Set up the model with default parameters
params = DLSParams.default()
model = DLSM(params)

# 2. Simulate a cohort of 10 students over 30 time steps
sim = Simulator(model, seed=42)
data = sim.simulate(n_students=10, n_timesteps=30)

# 3. Filter the first student's observations
pf = ParticleFilter(model, n_particles=500, seed=0)
result = pf.filter(
    observations=data.observations[0],
    events=data.events[0],
)

# 4. Inspect the result
print("Filtered states shape:", result.mean_states.shape)   # (30, 6)
print("Log-likelihood:      ", round(result.log_likelihood, 2))
print("Mean ESS:            ", round(result.ess.mean(), 1))
print("Final status:        ", result.mean_status[-1])

# 5. Compare with ground truth
true_engagement = data.states[0, :, :4]
est_engagement  = result.mean_states[:, :4]
rmse = np.sqrt(np.mean((true_engagement - est_engagement) ** 2))
print("Engagement RMSE:     ", round(rmse, 4))
```

That is the full workflow. Everything else in this guide elaborates on these five steps.

---

# Part II — Core Concepts

## 4. The Latent State

### 4.1 State vector

At each time \(t\), a student's latent state is:

\[
\mathbf{X}_t = \big( S_t,\; \mathbf{E}_t,\; P_t,\; R_t \big)
\]

| Component | Symbol | Shape | Range |
|-----------|--------|-------|-------|
| Status | \(S_t\) | scalar | \(\{0,1,2,3,4\}\) |
| Engagement | \(\mathbf{E}_t\) | vector of 4 | \([0,1]^4\) |
| Progress | \(P_t\) | scalar | \([0,1]\) |
| Risk | \(R_t\) | scalar | \([0,1]\) |

### 4.2 Engagement dimensions

The four engagement components are:

| Code | Name | Meaning | Example indicator |
|------|------|---------|-------------------|
| `eO` | Observing | Passive consumption | Page views, video plays, downloads |
| `eP` | Participating | Active involvement | Forum reads and replies, quiz attempts |
| `eC` | Contributing | Agentic contribution | Original forum posts, peer help, tutor questions |
| `eSR` | Self-Regulated | Strategic learning | Study planning, self-assessment, revision |

These map onto established engagement theory (Fredricks et al., 2004; Reeve & Tseng, 2011). The hierarchical constraint \(e^O \geq e^P \geq e^C\) reflects the observation that more intensive engagement typically requires a foundation of basic engagement. The library enforces this constraint via projection onto the ordered cone (Section 4.5).

### 4.3 Statuses

`Status` is an `IntEnum`:

```python
from dlsm.core.state import Status

Status.REGISTERED    # 0
Status.PROGRESSING   # 1
Status.STOPPED_OUT   # 2
Status.WITHDRAWN     # 3   <- absorbing
Status.COMPLETED     # 4   <- absorbing
```

`WITHDRAWN` and `COMPLETED` are absorbing: once a student enters, they cannot leave.

### 4.4 Risk

\(R_t\) is a continuous latent variable representing the propensity for adverse outcomes (withdrawal, prolonged stop-out). It is **not** the same as the discrete status:

- Status is observed (administrative records).
- Risk is latent and inferred from engagement, progress, and inactivity.
- Risk influences the *timing* of status transitions.
- A student can be "Progressing" with high risk, or "Stopped-Out" with low risk (likely to return).

### 4.5 The ordered-cone constraint

The constraint \(e^O \geq e^P \geq e^C\) is enforced by projecting onto the set:

\[
\big\{ \mathbf{e} \in [0,1]^4 : e^O \geq e^P \geq e^C \big\}
\]

You rarely need to call the projection directly, but it is exposed for advanced use:

```python
from dlsm.core.constraints import project_ordered_cone
import numpy as np

e = np.array([[0.5, 0.7, 0.3, 0.2]])   # violates the constraint
p = project_ordered_cone(e)
print(p)                                # [[0.6, 0.6, 0.3, 0.2]]
```

### 4.6 The DLSMState class

For single-state manipulation, use `DLSMState`:

```python
from dlsm.core.state import DLSMState, Status
import numpy as np

s = DLSMState(
    status=Status.PROGRESSING,
    continuous=np.array([0.6, 0.4, 0.2, 0.5, 0.3, 0.15]),
)
print(s.engagement)   # [0.6 0.4 0.2 0.5]
print(s.progress)     # 0.3
print(s.risk)         # 0.15
print(s.e_O)          # 0.6
```

---

## 5. Parameters and What They Mean

### 5.1 Creating a parameter set

```python
from dlsm import DLSParams

params = DLSParams.default()             # sensible defaults
params = DLSParams(alpha_O=0.4)          # override one field
params = DLSParams(
    alpha_O=0.35, lambda_O=0.12,
    gamma=0.25, omega_3=0.08,
)                                        # override several
```

### 5.2 The parameter groups

#### Engagement growth rates (\(\alpha\))

Control how fast engagement rises when the corresponding event occurs.

| Parameter | Effect |
|-----------|--------|
| `alpha_O` | How fast observing increases with access |
| `alpha_P` | How fast participating increases (also depends on `eO`) |
| `alpha_C` | How fast contributing increases (also depends on `eP`) |
| `alpha_SR` | How fast self-regulation increases with self-regulation events |

All are strictly positive. Typical range: 0.1 – 0.5.

#### Engagement decay rates (\(\lambda\))

Control how fast engagement decays in the absence of events.

| Parameter | Effect |
|-----------|--------|
| `lambda_O`, `lambda_P`, `lambda_C`, `lambda_SR` | Decay of the corresponding dimension |

All are strictly positive. Typical range: 0.05 – 0.3.

**Rule of thumb:** \(\alpha\) controls the ceiling; \(\lambda\) controls the floor. If \(\alpha \gg \lambda\), engagement rises quickly and stays high. If \(\lambda \gg \alpha\), engagement decays quickly.

#### Progress dynamics (\(\beta\))

Control how assessment completions translate into progress.

| Parameter | Meaning |
|-----------|---------|
| `beta_0` | Baseline log-odds of progressing |
| `beta_1` | Weight on contributing engagement \(e^C\) |
| `beta_2` | Weight on self-regulated engagement \(e^{SR}\) |
| `beta_3` | Weight on the assessment score |

Progress increments are \(w_a \cdot \sigma(\beta_0 + \beta_1 e^C + \beta_2 e^{SR} + \beta_3 \cdot \text{score})\).

#### Risk dynamics (\(\gamma, \omega\))

| Parameter | Meaning |
|-----------|---------|
| `gamma` | How fast risk moves toward its target |
| `omega_0` | Baseline log-odds of being at-risk |
| `omega_1` | Weight on mean engagement (negative effect) |
| `omega_2` | Weight on progress (negative effect) |
| `omega_3` | Weight on days inactive (positive effect) |

The risk target is:
\[
R^* = \sigma\big( \omega_0 - \omega_1 \bar{E} - \omega_2 P + \omega_3 \Delta t_{\text{inactive}} \big)
\]

#### Process noise (`q_diag`)

Length-6 vector of process noise variances, one per continuous component:

```python
params.q_diag = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.02])
```

Larger values mean more stochastic dynamics; the filter will track more loosely.

#### Status transition coefficients (`w_status`)

Shape `(5, 5, 6)`. `w_status[s, s', :]` is the coefficient vector for transitioning from status `s` to `s'`, with features `[eO, eP, eC, eSR, P, R]`.

You rarely need to set this by hand; the default is sensible. To customize:

```python
import numpy as np
params.w_status = np.zeros((5, 5, 6))
# Withdrawn and Completed are absorbing
for s in (3, 4):
    for s2 in range(5):
        if s2 != s:
            params.w_status[s, s2, :] = -20.0
```

#### Observation weight matrices

| Attribute | Shape | Purpose |
|-----------|-------|---------|
| `w_count` | `(K_count, 6)` | Poisson channel log-rates |
| `w_binary` | `(K_binary, 6)` | Bernoulli channel logits |
| `w_ordinal` | `(K_ordinal, 6)` | Ordered-probit channel logits |
| `ordinal_thresholds` | `(M-1,)` | Ordered-probit cut points |

### 5.3 Serializing parameters

```python
v = params.as_vector()                 # 1-D numpy array of 17 floats
print(v.shape)                         # (17,)

params2 = DLSParams.from_vector(v, template=params)
```

The 17-vector contains only the rate parameters (α, λ, β, γ, ω). Non-rate fields (matrices, thresholds) are taken from the template. This is what `PMCMC` uses internally.

### 5.4 The parameter names

```python
from dlsm.inference.pmcmc import PMCMC
print(PMCMC.PARAM_NAMES)
# ['alpha_O', 'alpha_P', 'alpha_C', 'alpha_SR',
#  'lambda_O', 'lambda_P', 'lambda_C', 'lambda_SR',
#  'beta_0', 'beta_1', 'beta_2', 'beta_3',
#  'gamma',
#  'omega_0', 'omega_1', 'omega_2', 'omega_3']
```

---

## 6. Observations and Measurement Channels

### 6.1 The Observation class

A single observation record at one time step:

```python
from dlsm.core.observations import Observation
import numpy as np

obs = Observation(
    counts=np.array([3, 1, 0, 2]),       # shape (K_count,)
    binaries=np.array([1]),              # shape (K_binary,)
    ordinals=np.array([2]),              # shape (K_ordinal,)
    events=np.array([1, 1, 0, 1]),       # shape (4,)
)
```

Any channel can be `None`, which means the observation is missing for that channel at that time step. This is how the library handles missing data: the particle filter simply skips the update for missing channels.

### 6.2 Observation channels

Three channel types are supported out of the box:

#### Poisson (counts)

Used for LMS activity counts. Rate is exponential in the latent state:

\[
y_k \sim \text{Poisson}\big( \exp(\mathbf{w}_k^\top \mathbf{X}) \big)
\]

Default: 4 channels, one per engagement dimension.

#### Bernoulli (binary)

Used for submission indicators, login flags, etc.:

\[
y_k \sim \text{Bernoulli}\big( \sigma(\mathbf{w}_k^\top \mathbf{X}) \big)
\]

Default: 1 channel, driven by progress.

#### Ordered probit (ordinal)

Used for self-reported engagement, Likert responses, etc.:

\[
y_k \sim \text{OrderedProbit}\big( \mathbf{w}_k^\top \mathbf{X}, \boldsymbol{\theta}_k \big)
\]

Default: 1 channel with 3 thresholds (4 levels), driven by self-regulated engagement.

### 6.3 The default channel layout

| Channel | Type | Drives on | Interpretation |
|---------|------|-----------|----------------|
| `counts[0]` | Poisson | \(e^O\) | Observing activity count |
| `counts[1]` | Poisson | \(e^P\) | Participating activity count |
| `counts[2]` | Poisson | \(e^C\) | Contributing activity count |
| `counts[3]` | Poisson | \(e^{SR}\) | Self-regulation activity count |
| `binaries[0]` | Bernoulli | \(P\) | Assessment submitted |
| `ordinals[0]` | Ordered probit | \(e^{SR}\) | Self-reported engagement |

You can redefine the layout by editing `w_count`, `w_binary`, `w_ordinal`, and `ordinal_thresholds`.

### 6.4 Missing data

Missing channels are handled automatically:

```python
obs_partial = Observation(
    counts=np.array([3, 1, 0, 2]),
    binaries=None,       # missing
    ordinals=None,       # missing
)
```

The particle filter will propagate the state forward without updating on the missing channels.

---

# Part III — Workflows

## 7. Simulating Data

### 7.1 Basic simulation

```python
from dlsm import DLSM, DLSParams, Simulator

model = DLSM(DLSParams.default())
sim = Simulator(model, seed=42)
data = sim.simulate(
    n_students=50,
    n_timesteps=60,
    event_prob=0.3,
)
```

### 7.2 What you get

```python
data.states           # (50, 60, 6)   continuous states
data.statuses         # (50, 60)      discrete statuses
data.observations     # list of 50 lists of 60 Observation objects
data.events           # (50, 60, 4)   binary event indicators
```

### 7.3 Custom initial state

```python
import numpy as np

x0 = np.array([0.3, 0.2, 0.1, 0.4, 0.0, 0.5])
data = sim.simulate(n_students=20, n_timesteps=60, x0=x0)
```

### 7.4 Custom event process

By default, events are i.i.d. Bernoulli with probability `event_prob`. To use a custom event process:

```python
rng = np.random.default_rng(0)
events = (rng.random((20, 60, 4)) < 0.4).astype(int)
events[:, :10, :] = 0    # first 10 steps: no events

# Use with model.simulate_trajectory directly
states, statuses, obs = model.simulate_trajectory(
    n_timesteps=60, events=events[0], rng=rng,
)
```

### 7.5 Simulating a specific scenario

To simulate a "disengaging student", start with high engagement and set `event_prob` low:

```python
x0_engaged = np.array([0.8, 0.6, 0.4, 0.7, 0.3, 0.1])
data = sim.simulate(
    n_students=1, n_timesteps=90,
    event_prob=0.05, x0=x0_engaged,
)
# Engagement decays over time
```

---

## 8. Filtering: Tracking Latent States

### 8.1 Basic filtering

```python
from dlsm import DLSM, DLSParams, ParticleFilter

model = DLSM(DLSParams.default())
pf = ParticleFilter(model, n_particles=1000, seed=0)

result = pf.filter(
    observations=data.observations[0],
    events=data.events[0],
)
```

### 8.2 The FilterResult object

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `mean_states` | `(T, 6)` | Posterior mean of the continuous state |
| `var_states` | `(T, 6)` | Posterior variance |
| `status_probs` | `(T, 5)` | Posterior probability of each status |
| `mean_status` | `(T,)` | Most likely status at each time |
| `log_likelihood` | scalar | Estimate of \(\log p(y_{1:T})\) |
| `ess` | `(T,)` | Effective sample size per step |
| `particles` | `(T, N, 6)` or `None` | Full particle trajectories (if requested) |
| `weights` | `(T, N)` or `None` | Particle weights (if requested) |

### 8.3 Computing credible intervals

```python
import numpy as np

mean = result.mean_states          # (T, 6)
std  = np.sqrt(result.var_states)  # (T, 6)

lower = mean - 1.96 * std
upper = mean + 1.96 * std
```

### 8.4 Storing full particles

For downstream analysis (e.g., custom quantiles), request the full particle set:

```python
result = pf.filter(
    observations=data.observations[0],
    events=data.events[0],
    store_particles=True,
)

particles = result.particles      # (T, N, 6)
weights   = result.weights        # (T, N)

# Custom quantile at time t=30
t = 30
idx = np.argsort(weights[t])
cum = np.cumsum(weights[t][idx])
p_10 = particles[t, idx[np.searchsorted(cum, 0.10)], 5]   # risk quantile
p_90 = particles[t, idx[np.searchsorted(cum, 0.90)], 5]
```

### 8.5 Choosing the number of particles

| `n_particles` | Use case | Runtime per pass |
|---------------|----------|------------------|
| 100 | Quick prototype | ~0.02 s |
| 500 | Interactive analysis | ~0.07 s |
| 1,000 | Standard filtering | ~0.13 s |
| 2,000 | PMCMC inner loop | ~0.25 s |
| 5,000 | High-accuracy posterior | ~0.62 s |

(Runtimes measured on a 2.4 GHz Intel i7 with `T = 100`.)

Rule of thumb: start with 1,000. If `mean(ess)` is much smaller than `n_particles / 2`, increase.

### 8.6 Likelihood-only mode

For model comparison or MCMC, you often only need the log-likelihood:

```python
ll = pf.estimate_log_likelihood(
    observations=data.observations[0],
    events=data.events[0],
)
print(ll)
```

---

## 9. Estimating Parameters with PMCMC

### 9.1 Basic workflow

```python
from dlsm import DLSM, DLSParams, PMCMC

model = DLSM(DLSParams.default())
pmcmc = PMCMC(
    model,
    n_particles=2000,
    n_iterations=10000,
    burn_in=2000,
    thin=10,
    proposal_scale=0.02,
    seed=0,
)

result = pmcmc.fit(
    observations=data.observations[0],
    events=data.events[0],
)
```

### 9.2 The PMCMCResult object

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `chain` | `(n_keep, 17)` | Posterior samples after burn-in and thinning |
| `log_posterior` | `(n_keep,)` | Log-posterior at each retained sample |
| `acceptance_rate` | scalar | Fraction of proposals accepted |
| `param_names` | list of str | Names of the 17 parameters |

Convenience methods:

```python
result.posterior_mean()                # (17,)
result.posterior_std()                 # (17,)
result.credible_interval(level=0.95)   # (2, 17)
```

### 9.3 Reading the output

```python
from dlsm.evaluation import metrics

table = metrics.parameter_recovery_table(
    true_params=DLSParams.default().as_vector(),
    posterior_samples=result.chain,
    param_names=result.param_names,
)
for row in table:
    print(
        f"{row['parameter']:10s}  "
        f"true={row['true']:+.3f}  "
        f"post={row['posterior_mean']:+.3f}  "
        f"std={row['posterior_std']:.3f}  "
        f"covered={row['covered']}"
    )
```

### 9.4 Tuning the proposal scale

The `proposal_scale` controls the standard deviation of the Gaussian random-walk proposal. Target acceptance rate: 0.2 – 0.4.

| Acceptance rate | Action |
|-----------------|--------|
| > 0.5 | Too small scale; increase |
| 0.2 – 0.4 | Well-tuned |
| < 0.15 | Too large scale; decrease |

Per-parameter scaling is also supported:

```python
scales = np.full(17, 0.02)
scales[0:4] = 0.03       # slightly larger for α
scales[4:8] = 0.015      # smaller for λ

pmcmc = PMCMC(model, proposal_scale=scales, ...)
```

### 9.5 Multi-chain runs

For reliable convergence diagnostics, run multiple chains with different seeds:

```python
chains = []
for seed in (0, 1, 2):
    pmcmc = PMCMC(model, seed=seed, ...)
    chains.append(pmcmc.fit(obs, events).chain)

stacked = np.stack(chains)     # (3, n_keep, 17)

from dlsm.inference.diagnostics import gelman_rubin, effective_sample_size
rhat = gelman_rubin(stacked)
ess  = effective_sample_size(stacked)

print("Max R-hat:", rhat.max())          # target < 1.1
print("Min ESS:  ", ess.min())           # target > 500
```

### 9.6 Computational cost

PMCMC is expensive: each iteration runs a full particle filter. Approximate cost:

\[
\text{cost} \approx N_{\text{iterations}} \times N_{\text{particles}} \times T \times C
\]

For \(N_{\text{iterations}} = 10000\), \(N_{\text{particles}} = 2000\), \(T = 60\): on the order of hours on a laptop, minutes on a multi-core server. For exploration, use smaller settings:

```python
pmcmc = PMCMC(model, n_particles=300, n_iterations=1000, burn_in=200, thin=2)
```

Once the acceptance rate is tuned, increase particles and iterations.

---

## 10. Working with Real Data

### 10.1 Data requirements

To apply the DLSM to real data, you need per-student time series of:

1. **Registration and status records** — dates and status codes.
2. **LMS activity counts** — daily or weekly counts of views, downloads, plays.
3. **Assessment records** — submission dates, scores.
4. **Communication logs** — tutor contacts, forum posts.
5. **Extension requests** — dates and outcomes.

### 10.2 Building Observation objects

```python
from dlsm.core.observations import Observation
import numpy as np
import pandas as pd

def build_observations(df: pd.DataFrame) -> list[Observation]:
    """
    df must have columns:
        time, views, downloads, plays, submissions, tutor_emails, self_report
    """
    obs_list = []
    for _, row in df.iterrows():
        obs = Observation(
            counts=np.array([
                row["views"] + row["downloads"] + row["plays"],
                0,                               # participating (placeholder)
                0,                               # contributing (placeholder)
                row["tutor_emails"],             # proxy for self-regulated
            ]),
            binaries=np.array([int(row["submissions"] > 0)]),
            ordinals=np.array([row["self_report"]]) if pd.notna(row["self_report"]) else None,
        )
        obs_list.append(obs)
    return obs_list
```

You will need to design the mapping from your raw logs to the six channels. Section 11 explains how to customize the channel layout to match your data.

### 10.3 Aligning time

Self-paced learners have irregular timestamps. Two strategies:

**Strategy A — Uniform grid.** Bin all activity into fixed windows (e.g., one week). Missing windows become observations with all-zero counts (which is informative, not missing).

```python
def resample_to_weekly(df, student_id):
    df = df.set_index("timestamp")
    weekly = df.resample("W").agg({
        "views": "sum",
        "downloads": "sum",
        "plays": "sum",
        "submissions": "sum",
        "tutor_emails": "sum",
    }).fillna(0)
    return weekly
```

**Strategy B — Event-driven.** Keep the irregular timestamps and pass them to the filter as `dt` values. The particle filter's `step` accepts arbitrary `dt`, so you can propagate the state by the actual elapsed time between observations:

```python
for t in range(1, len(obs_list)):
    dt = (timestamps[t] - timestamps[t-1]).days
    # the filter internally uses dt for the drift
```

For simplicity, start with Strategy A.

### 10.4 Filtering a new student

Once the model is fitted:

```python
pf = ParticleFilter(fitted_model, n_particles=1000, seed=0)
result = pf.filter(
    observations=new_student_obs,
    events=new_student_events,
)

# Real-time risk estimate at the latest time
latest_risk = result.mean_states[-1, 5]
latest_risk_std = np.sqrt(result.var_states[-1, 5])
print(f"Risk: {latest_risk:.2f} ± {latest_risk_std:.2f}")
```

### 10.5 Batch filtering across a cohort

```python
from tqdm import tqdm

def filter_cohort(model, data, n_particles=500, seed=0):
    results = []
    for i in tqdm(range(len(data.observations)), desc="Filtering"):
        pf = ParticleFilter(model, n_particles=n_particles, seed=seed + i)
        r = pf.filter(data.observations[i], events=data.events[i])
        results.append(r)
    return results
```

Parallelization:

```python
from joblib import Parallel, delayed

def _one(i, model, data, seed):
    pf = ParticleFilter(model, n_particles=500, seed=seed + i)
    return pf.filter(data.observations[i], events=data.events[i])

results = Parallel(n_jobs=-1)(
    delayed(_one)(i, model, data, seed=0)
    for i in range(len(data.observations))
)
```

---

# Part IV — Advanced Topics

## 11. Custom Observation Channels

### 11.1 Redefining the channel layout

Suppose your data has:
- Two LMS count channels: `page_views` and `video_plays`.
- One binary: `assignment_submitted`.
- One ordinal: `self_reported_engagement`.

You want `page_views` to load on \(e^O\), `video_plays` to load on both \(e^O\) and \(e^P\):

```python
import numpy as np
from dlsm import DLSParams

params = DLSParams.default()

# Rows: channels. Columns: [eO, eP, eC, eSR, P, R]
params.w_count = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # page_views -> eO
    [0.7, 0.5, 0.0, 0.0, 0.0, 0.0],   # video_plays -> eO + eP
])

params.w_binary = np.array([
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],   # submitted -> P
])

params.w_ordinal = np.array([
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],   # self-report -> eSR
])

params.ordinal_thresholds = np.array([-0.5, 0.5, 1.5])   # 4 levels
```

Now `Observation` objects should provide arrays of the matching shape:

```python
obs = Observation(
    counts=np.array([page_views, video_plays]),        # shape (2,)
    binaries=np.array([submitted]),                    # shape (1,)
    ordinals=np.array([self_report]),                  # shape (1,)
)
```

### 11.2 Adding a new channel type

Subclass `ObservationModel` to add, e.g., a Gaussian channel:

```python
from dlsm.core.observations import ObservationModel
import numpy as np

class ExtendedObservationModel(ObservationModel):

    def __init__(self, params, w_gauss=None, sigma_gauss=0.3):
        super().__init__(params)
        self.w_gauss = w_gauss if w_gauss is not None else np.zeros((1, 6))
        self.sigma_gauss = sigma_gauss

    def log_likelihood_gaussian(self, x, y):
        mu = x @ self.w_gauss.T
        resid = y - mu
        return -0.5 * (resid / self.sigma_gauss) ** 2 \
               - np.log(self.sigma_gauss) \
               - 0.5 * np.log(2 * np.pi)

    def log_likelihood(self, x, counts=None, binaries=None, ordinals=None, gaussian=None):
        ll = super().log_likelihood(x, counts, binaries, ordinals)
        if gaussian is not None:
            ll = ll + self.log_likelihood_gaussian(x, gaussian).sum(axis=1)
        return ll

    def sample(self, x, rng=None):
        obs = super().sample(x, rng=rng)
        if self.w_gauss is not None:
            mu = x @ self.w_gauss.T
            obs_gauss = mu + (rng or np.random.default_rng()).normal(
                0, self.sigma_gauss, size=mu.shape,
            )
            obs.__dict__["gaussian"] = obs_gauss
        return obs
```

Then attach it to a `DLSM`:

```python
from dlsm import DLSM

model = DLSM(params)
model.observation = ExtendedObservationModel(params)
```

## 12. Custom Drift Functions

### 12.1 Modifying the engagement drift

Subclass `Dynamics` to change the drift:

```python
from dlsm.core.dynamics import Dynamics
import numpy as np

class SaturatedDynamics(Dynamics):

    def drift(self, x, events, dt=None):
        base = super().drift(x, events, dt=dt)
        # Add a saturating term that pulls eO back toward 0.6
        base[:, 0] += -0.1 * (x[:, 0] - 0.6)
        return base

model = DLSM(params)
model.dynamics = SaturatedDynamics(params)
```

### 12.2 Adding a course-level random effect

```python
class RandomEffectDynamics(Dynamics):

    def __init__(self, params, course_effect=0.0):
        super().__init__(params)
        self.course_effect = course_effect

    def drift(self, x, events, dt=None):
        base = super().drift(x, events, dt=dt)
        # Shift all engagement drift by a course-level offset
        base[:, :4] += self.course_effect
        return base
```

### 12.3 Time-varying dt

The `step` method accepts a `dt` argument for irregularly-spaced observations:

```python
dt = 5.0    # five days elapsed since last observation
x_next = dynamics.step(x, events, dt=dt, rng=rng)
```

The drift is scaled by `dt` and the process noise by \(\sqrt{dt}\), preserving the continuous-time interpretation.

---

## 13. Diagnostics and Model Checking

### 13.1 Particle-filter diagnostics

#### Effective sample size

```python
result = pf.filter(obs, events=events)
print("Mean ESS:   ", result.ess.mean())
print("Min ESS:    ", result.ess.min())
print("Below 30%:  ", (result.ess < 0.3 * pf.n_particles).mean())
```

If `mean(ess) < 0.3 * n_particles`, increase `n_particles` or investigate the model.

#### Weight degeneracy check

```python
weights = result.weights[-1]
print("Max weight: ", weights.max())          # should be << 1
print("Weight entropy:", -(weights * np.log(weights + 1e-12)).sum())
```

High max weight (e.g., > 0.1) indicates degeneracy.

### 13.2 PMCMC diagnostics

#### Trace plots

```python
from dlsm.viz import plots

fig = plots.plot_posterior(
    result.chain,
    result.param_names,
    true_params=params.as_vector(),
)
fig.savefig("posteriors.png", dpi=150)
```

#### Gelman-Rubin (\(\hat{R}\))

```python
from dlsm.inference.diagnostics import gelman_rubin

# Requires ≥ 2 chains
rhat = gelman_rubin(np.stack([chain1, chain2, chain3]))
print("R-hat:", rhat)
# R-hat < 1.1 for all parameters indicates convergence
```

#### Effective sample size

```python
from dlsm.inference.diagnostics import effective_sample_size

ess = effective_sample_size(np.stack([chain1, chain2]))
print("ESS:", ess)
# Aim for ESS > 500 per parameter
```

### 13.3 Posterior predictive checks

Simulate from the fitted model and compare to observed data:

```python
from dlsm import DLSM, DLSParams

# Fit parameters (obtain posterior mean)
theta_hat = result.posterior_mean()
params_hat = DLSParams.from_vector(theta_hat, template=params)
model_hat = DLSM(params_hat)

# Simulate from the fitted model
from dlsm import Simulator
sim = Simulator(model_hat, seed=0)
sim_data = sim.simulate(n_students=50, n_timesteps=60)

# Compare summary statistics (e.g., mean engagement)
print("Observed mean eO:", data.states[..., 0].mean())
print("Simulated mean eO:", sim_data.states[..., 0].mean())
```

### 13.4 Residual analysis

```python
# Compare filtered states to observed counts
filtered_counts = np.exp(result.mean_states @ params.w_count.T)
observed_counts = np.stack([
    o.counts for o in data.observations[0]
])
residuals = observed_counts - filtered_counts
print("Mean residual:  ", residuals.mean(axis=0))
print("Std residual:   ", residuals.std(axis=0))
```

Systematic residual structure indicates model misspecification.

---

## 14. Performance and Scaling

### 14.1 Profiling a single filtering pass

```python
import time

t0 = time.perf_counter()
pf.filter(obs, events=events)
elapsed = time.perf_counter() - t0
print(f"Filtering: {elapsed:.3f} s")
```

### 14.2 Vectorization

The particle filter is fully vectorized across particles: `N_p = 1000` particles cost roughly the same per step as `N_p = 100` (about a 2–3× factor, not 10×), because NumPy handles the array arithmetic efficiently.

### 14.3 Parallelizing across students

Filtering is embarrassingly parallel across students:

```python
from joblib import Parallel, delayed

def _filter_one(i, obs, events, model, seed):
    pf = ParticleFilter(model, n_particles=500, seed=seed + i)
    return pf.filter(obs, events=events)

results = Parallel(n_jobs=-1)(
    delayed(_filter_one)(i, data.observations[i], data.events[i], model, 0)
    for i in range(len(data.observations))
)
```

### 14.4 Parallelizing PMCMC

PMCMC is sequential within a chain, but independent across chains:

```python
def _chain(seed, obs, events, model):
    pmcmc = PMCMC(model, seed=seed, n_particles=500, n_iterations=2000)
    return pmcmc.fit(obs, events).chain

chains = Parallel(n_jobs=-1)(
    delayed(_chain)(seed, obs, events, model) for seed in range(4)
)
```

### 14.5 Distributed particle filtering

For very large cohorts (> 10⁴ students), distribute across machines:

- Partition students across worker processes.
- Each worker runs a full `ParticleFilter` on its shard.
- Aggregate results on the driver.

The library does not ship a distributed scheduler, but the API composes cleanly with `dask`, `ray`, or `joblib`.

### 14.6 Memory considerations

For a trajectory of `T = 100` with `N_p = 1000` particles:

- Without `store_particles`: ~50 MB.
- With `store_particles=True`: ~480 MB (stores all `T × N × 6` floats).

Only request stored particles when you need them.

---

## 15. Reproducibility

### 15.1 Seeds everywhere

Every stochastic component accepts a seed:

```python
sim = Simulator(model, seed=42)
pf = ParticleFilter(model, n_particles=1000, seed=0)
pmcmc = PMCMC(model, seed=0, ...)
```

Same seed → identical results:

```python
a = Simulator(model, seed=42).simulate(n_students=5, n_timesteps=20)
b = Simulator(model, seed=42).simulate(n_students=5, n_timesteps=20)
assert np.allclose(a.states, b.states)
```

### 15.2 Locking the environment

For published work, pin dependencies:

```bash
pip install -r requirements-lock.txt
```

The lock file used for the paper's simulation study is included in the repository.

### 15.3 Logging a run

Best practice for empirical work:

```python
import json
from datetime import datetime

run_metadata = {
    "date": datetime.now().isoformat(),
    "model_params": params.to_dict(),
    "n_particles": pf.n_particles,
    "seed": pf.rng.bit_generator.state,    # optional, for full reproducibility
    "n_students": len(data.observations),
    "n_timesteps": len(data.observations[0]),
}
with open("run_metadata.json", "w") as f:
    json.dump(run_metadata, f, indent=2)
```

---

# Part V — Reference

## 16. API Cheat Sheet

### Core

```python
from dlsm import (
    DLSParams,        # parameter container
    DLSM,             # model
    DLSMState,        # single state
    Status,           # discrete status enum
    Dynamics,         # transition dynamics
    ObservationModel, # observation channels
    Observation,      # single observation
)
```

### Inference

```python
from dlsm import ParticleFilter, PMCMC
from dlsm.inference import (
    gelman_rubin,
    effective_sample_size,
)
```

### Simulation

```python
from dlsm import Simulator
```

### Evaluation & visualization

```python
from dlsm.evaluation import metrics
from dlsm.viz import plots
```

### Common patterns

| Task | Code |
|------|------|
| Create default model | `DLSM(DLSParams.default())` |
| Simulate cohort | `Simulator(model).simulate(50, 60)` |
| Filter one student | `ParticleFilter(model).filter(obs, events)` |
| Estimate parameters | `PMCMC(model).fit(obs, events)` |
| Compute RMSE | `metrics.rmse(true, pred)` |
| Compute coverage | `metrics.coverage(true, lo, hi)` |
| Plot trajectories | `plots.plot_latent_trajectories(...)` |

### The 17-vector

```python
v = params.as_vector()
# v[0:4]    α (alpha_O, alpha_P, alpha_C, alpha_SR)
# v[4:8]    λ (lambda_O, lambda_P, lambda_C, lambda_SR)
# v[8:12]   β (beta_0, beta_1, beta_2, beta_3)
# v[12]     γ (gamma)
# v[13:17]  ω (omega_0, omega_1, omega_2, omega_3)
```

### The 6-vector (continuous state)

```python
x = state.continuous
# x[0]  eO
# x[1]  eP
# x[2]  eC
# x[3]  eSR
# x[4]  P
# x[5]  R
```

---

## 17. Troubleshooting and FAQ

### Q1: The particle filter ESS collapses to 1 immediately

**Symptom:** `result.ess` drops to 1 after the first few time steps.

**Causes and fixes:**
- **Too few particles** — increase `n_particles` (try 2000+).
- **Observation model mismatch** — your data has much higher counts than the model expects. Rescale counts or adjust `w_count`.
- **Process noise too small** — increase `q_diag` so the filter can explore.

### Q2: PMCMC acceptance rate is near 0

**Fix:** Decrease `proposal_scale`:

```python
pmcmc = PMCMC(model, proposal_scale=0.005, ...)
```

Target 0.2 – 0.4.

### Q3: PMCMC acceptance rate is near 1

**Fix:** Increase `proposal_scale`:

```python
pmcmc = PMCMC(model, proposal_scale=0.05, ...)
```

### Q4: Parameter recovery is poor for some parameters

**Diagnosis:** Check posterior correlations.

```python
import numpy as np
corr = np.corrcoef(result.chain, rowvar=False)
# Look for |corr| > 0.8 pairs
```

**Fix:** Add identifying constraints, or fix one parameter to a known value. See Section 5.2 of the manuscript on identifiability.

### Q5: The ordered-cone projection changes my estimates too much

**Diagnosis:** Check how often projection is active:

```python
raw = result.mean_states[:, :3]
from dlsm.core.constraints import project_ordered_cone
projected = project_ordered_cone(raw)
violations = np.abs(raw - projected).sum(axis=1) > 1e-6
print(f"Projection active at {violations.mean():.1%} of time steps")
```

If projection is active most of the time, either:
- The constraint may not fit your data (Section 3.6 of the manuscript discusses this).
- The engagement model needs refitting.

### Q6: Simulation is too slow for large cohorts

**Fix:** Use smaller `n_timesteps` for prototyping, then scale up. Also consider generating the event matrix once and reusing it across conditions.

### Q7: I have missing timestamps in my data

**Fix:** Fill missing time steps with all-zero observations:

```python
obs = Observation(
    counts=np.zeros(K_count, dtype=int),
    binaries=np.zeros(K_binary, dtype=int),
    ordinals=None,           # truly missing, not zero
)
```

Zero counts are informative (the student was inactive); missing ordinals are not.

### Q8: `ValueError: q_diag must have length 6`

**Fix:** `q_diag` is the process noise variance, one per continuous component. Always length 6:

```python
params.q_diag = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.02])
```

### Q9: `ValueError: w_status must have shape (5, 5, 6)`

**Fix:** `w_status[s, s', :]` is the coefficient vector for transitioning from status `s` to `s'`, with 6 features. Use the default as a template:

```python
from dlsm.core.parameters import _default_status_coefficients
params.w_status = _default_status_coefficients()
```

### Q10: How do I know if my model is well-specified?

**Checklist:**
1. **ESS stays high** — no degeneracy.
2. **Posterior predictive checks match** — simulated data resembles observed.
3. **Residuals are unstructured** — no systematic patterns.
4. **PMCMC converges** — $\hat{R} < 1.1$.
5. **Parameters are identified** — no extreme posterior correlations.

---

## 18. Best Practices

### 18.1 Model development

1. **Start with the default parameters.** Fit them to your data before modifying the model.
2. **Prototype with simulation.** Verify the model recovers what you put in before applying to real data.
3. **Inspect the posterior.** Never trust a single point estimate; always look at the chain.
4. **Document your channel mapping.** Which observation maps to which latent dimension is a modeling choice; record it.

### 18.2 Inference

1. **Always check ESS.** If ESS is low, increase particles before drawing conclusions.
2. **Use multiple chains for PMCMC.** A single chain cannot diagnose convergence.
3. **Tune the proposal scale.** Target 20–40% acceptance.
4. **Thin the chain.** Retain every 10th sample after burn-in to reduce autocorrelation.

### 18.3 Evaluation

1. **Report calibration, not just accuracy.** Brier score and calibration plots reveal more than AUC alone.
2. **Compare against static baselines.** Logistic regression and random forest are strong baselines.
3. **Held-out evaluation.** Train on some students, test on others — never evaluate on training data.
4. **Report uncertainty.** Always report credible intervals alongside point estimates.

### 18.4 Deployment

1. **Version your model.** Save `params.to_dict()` alongside any deployed predictions.
2. **Log the seed.** Even for production, log the RNG seed for incident debugging.
3. **Monitor ESS in production.** If ESS collapses, downstream predictions are unreliable.
4. **Warm-start with the last posterior.** For real-time tracking, reuse the last particle cloud as the prior for the next step rather than re-initializing.

### 18.5 Sharing

1. **Pin dependencies.** Publish a `requirements-lock.txt`.
2. **Publish code.** The library is designed to be reproducible; make it easy for others to verify.
3. **Cite both the paper and the software.** See `README.md` for BibTeX entries.
4. **Share diagnostics.** Convergence plots, ESS plots, and posterior plots make your results auditable.

---

## Where to Go Next

- **Manuscript:** *A Dynamic Latent-State Model of Student Engagement in Self-Paced Learning* — for the theoretical foundations and simulation study design.
- **README:** [github.com/hongxueharriswang/dlsm-lib](https://github.com/hongxueharriswang/dlsm-lib) — quick start, install, citation.
- **Examples:** `examples/` directory — runnable scripts for simulation, parameter recovery, and empirical validation.
- **Tests:** `tests/` directory — read the tests to see the API in action.
- **Issue tracker:** [github.com/hongxueharriswang/dlsm-lib/issues](https://github.com/hongxueharriswang/dlsm-lib/issues) — for bugs and feature requests.

---

**Last updated:** 2026-10-02 · **Version:** 0.1.0
# DLSM Tutorial

**A hands-on, progressive introduction to Dynamic Latent-State Modeling**

Version 0.1.0 · [github.com/hongxueharriswang/dlsm-lib](https://github.com/hongxueharriswang/dlsm-lib)

---

## How to Use This Tutorial

This tutorial is designed to be **read with a Python interpreter open**. Every code block is runnable, and each section ends with a **checkpoint** that tells you what you should have learned and what you should be able to do.

The tutorial has **six chapters**, each building on the previous:

| Chapter | Focus | Time | Prerequisites |
|---------|-------|------|---------------|
| 1 | Meet the model: state, parameters, dynamics | 15 min | Python, NumPy |
| 2 | Simulate a single student | 20 min | Chapter 1 |
| 3 | Track a student with the particle filter | 25 min | Chapter 2 |
| 4 | Estimate parameters with PMCMC | 30 min | Chapter 3 |
| 5 | Apply to a cohort and evaluate | 30 min | Chapter 4 |
| 6 | Extend the model for your own data | 40 min | Chapter 5 |

If you are short on time, Chapters 1–3 give you a working understanding. If you plan to use the library in research, work through all six.

**Setup before you begin:**

```bash
pip install "dlsm[all]"
```

```python
import numpy as np
import matplotlib.pyplot as plt
from dlsm import (
    DLSParams, DLSM, DLSMState, Status,
    Simulator, ParticleFilter, PMCMC,
    plots,
)
from dlsm.evaluation import metrics
from dlsm.core.observations import Observation

np.set_printoptions(precision=3, suppress=True)
```

---

# Chapter 1 — Meet the Model

**Learning objectives:**
- Understand what a "latent state" is and why it matters for engagement.
- Create and inspect a `DLSParams` object.
- Understand the 6-dimensional continuous state vector.
- Run a single transition of the dynamics.

---

## 1.1 The Story

Imagine a student, Amina, enrolled in a self-paced course at an open university. She registered in September. Some weeks she logs in daily and posts on the forum; other weeks she does nothing visible on the LMS but is reading a textbook at home. In November she requests an extension. In December she submits two assignments back-to-back. In January she goes silent for three weeks, then returns.

What is Amina's "engagement"?

Traditional learning analytics would compute her weekly login count, average session duration, and submission count, then feed those numbers into a classifier. But those numbers don't capture what's really happening: Amina's *underlying engagement* — her motivation, attention, and investment — is a continuous quantity that rises and falls, and it *generates* the observable behavior, sometimes with a delay and sometimes not at all.

The DLSM makes this intuition formal. Engagement is a **latent state** — not directly observable, but inferable from noisy observations. And it evolves over time according to dynamics we can write down.

## 1.2 The State

Let us set up a model and inspect its state:

```python
params = DLSParams.default()
model = DLSM(params)
print(type(model))
# <class 'dlsm.models.dlsm.DLSM'>
```

At any time \(t\), the student's latent state has **six continuous components** plus **one discrete status**:

```python
# A snapshot of Amina's state in week 10:
x = np.array([0.6, 0.35, 0.15, 0.5, 0.25, 0.4])
#           eO    eP     eC    eSR    P     R
```

Let's unpack each:

| Index | Symbol | Name | Meaning |
|-------|--------|------|---------|
| 0 | \(e^O\) | Observing | Passive consumption: reads, video plays |
| 1 | \(e^P\) | Participating | Active involvement: forum reads, quiz attempts |
| 2 | \(e^C\) | Contributing | Agentic contribution: original posts, peer help |
| 3 | \(e^{SR}\) | Self-Regulated | Strategic learning: planning, self-assessment |
| 4 | \(P\) | Progress | Academic progress through the course |
| 5 | \(R\) | Risk | Latent propensity for adverse outcomes |

Amina's snapshot says: she observes a lot (0.6), participates moderately (0.35), contributes little (0.15), self-regulates substantially (0.5), has completed a quarter of the course (0.25), and has moderate risk (0.4).

The discrete status is:

```python
status = Status.PROGRESSING
print(int(status))    # 1
```

There are five statuses: `REGISTERED=0`, `PROGRESSING=1`, `STOPPED_OUT=2`, `WITHDRAWN=3`, `COMPLETED=4`. The last two are **absorbing**: once a student enters, they never leave.

## 1.3 The `DLSMState` Class

For convenience, the library wraps a single state in a `DLSMState`:

```python
s = DLSMState(
    status=Status.PROGRESSING,
    continuous=np.array([0.6, 0.35, 0.15, 0.5, 0.25, 0.4]),
)
print(s.engagement)   # [0.6  0.35 0.15 0.5 ]
print(s.progress)     # 0.25
print(s.risk)         # 0.4
print(s.e_O)          # 0.6
print(s.e_SR)         # 0.5
```

This class also validates the state: length must be 6, status must be 0–4.

## 1.4 Parameters: What Shape the World

The `DLSParams` object holds every number that determines how the model behaves. Let's inspect the defaults:

```python
p = DLSParams.default()

# Engagement growth and decay
print("Growth rates (α):", p.alpha_O, p.alpha_P, p.alpha_C, p.alpha_SR)
# 0.3 0.25 0.2 0.35

print("Decay rates (λ): ", p.lambda_O, p.lambda_P, p.lambda_C, p.lambda_SR)
# 0.15 0.2 0.25 0.1
```

**How to read these numbers.** When Amina accesses the LMS (an "observing event"), her observing engagement \(e^O\) grows by \(\alpha_O (1 - e^O)\). The \((1 - e^O)\) factor caps growth: as \(e^O\) approaches 1, further events have less effect. Without any access events, \(e^O\) decays toward zero at rate \(\lambda_O\). So:

- **High α, low λ**: engagement rises quickly and stays high.
- **Low α, high λ**: engagement rises slowly and decays fast.
- **Default**: moderate — engagement rises within a week or two of regular activity, and decays over a month of silence.

Risk dynamics:

```python
print("Risk rate γ:     ", p.gamma)         # 0.2
print("Risk baseline ω₀:", p.omega_0)      # 1.5
print("Engagement ω₁:  ", p.omega_1)      # 2.0
print("Progress ω₂:    ", p.omega_2)      # 1.5
print("Inactivity ω₃:  ", p.omega_3)      # 0.05
```

Risk moves toward a target \(R^*\) at rate \(\gamma\). The target is high when engagement and progress are low, and when the student has been inactive for a long time. This captures the intuition that a disengaging, stalled student becomes at-risk.

## 1.5 One Step of the Dynamics

Let's advance Amina's state by one week with a single "access" event:

```python
dynamics = model.dynamics

# Amina accesses the LMS this week (only the "observing" event fires)
events = np.array([1, 0, 0, 0])

x_t = np.array([0.6, 0.35, 0.15, 0.5, 0.25, 0.4])
x_next = dynamics.step(
    x_t[None, :],
    events[None, :],
    rng=np.random.default_rng(0),
    add_noise=False,     # deterministic for clarity
)
print("Before:", x_t)
print("After: ", x_next[0])
```

You should see \(e^O\) increase (access event) and \(e^P, e^C, e^{SR}\) decrease slightly (decay, no participation/contribution/self-regulation). Progress is unchanged (no assessment). Risk moves slightly.

Now let's try a "productive week" where the student accesses, participates, and self-regulates:

```python
events = np.array([1, 1, 0, 1])
x_next = dynamics.step(
    x_t[None, :], events[None, :],
    rng=np.random.default_rng(0), add_noise=False,
)
print("Before:", x_t)
print("After: ", x_next[0])
```

Engagement rises across three dimensions. Note the cascading effect: participating requires observing as a precondition (in the drift equation, \(de^P/dt\) is proportional to \(e^O\)), so the two rise together.

## 1.6 The Ordered-Cones Constraint

The library enforces \(e^O \geq e^P \geq e^C\) after every step. Why? Because the theory says: you can't contribute without participating, and you can't participate without observing. The constraint is enforced by projection onto the ordered cone:

```python
from dlsm.core.constraints import project_ordered_cone

# A state that violates the constraint
bad = np.array([[0.3, 0.7, 0.5, 0.2]])
fixed = project_ordered_cone(bad)
print("Bad:  ", bad[0])
print("Fixed:", fixed[0])
# Bad:   [0.3 0.7 0.5 0.2]
# Fixed: [0.5 0.5 0.5 0.2]
```

The projection finds the closest valid vector in Euclidean distance. The fourth component (self-regulated) is unconstrained.

## 1.7 Checkpoint

You should now be able to:
- [x] Describe the state as 6 continuous + 1 discrete.
- [x] Name the four engagement dimensions.
- [x] Explain what α, λ, γ, and ω mean.
- [x] Run a single step of the dynamics.

**Try it yourself:** With the default parameters, start from `x = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]` and apply 10 consecutive weeks with no events. What happens to the engagement vector? What happens to risk?

<details>
<summary>Answer</summary>

```python
x = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
events = np.zeros((1, 4))
for t in range(10):
    x = dynamics.step(x, events, rng=np.random.default_rng(0), add_noise=False)
print(x[0])
# Engagement decays toward 0; risk rises toward its target
# ~[0.11 0.02 0.004 0.17 0.5 0.5]
```
</details>

---

# Chapter 2 — Simulate a Single Student

**Learning objectives:**
- Use `Simulator` to generate a realistic trajectory.
- Understand the observation model.
- Visualize a simulated trajectory.
- Manipulate the simulation to explore scenarios.

---

## 2.1 Simulating Amina

Let's simulate Amina's semester:

```python
sim = Simulator(model, seed=42)
states, statuses, observations = model.simulate_trajectory(
    n_timesteps=60,
    events=None,          # random events
    rng=np.random.default_rng(0),
)
print("States:", states.shape)        # (60, 6)
print("Statuses:", statuses.shape)    # (60,)
print("Observations:", len(observations))  # 60
```

By default, the initial state is `[0.1, 0.05, 0.01, 0.2, 0.0, 0.8]`: a newly registered student with low engagement, zero progress, and high risk (because they haven't started yet).

## 2.2 What Does an Observation Look Like?

At each time step, the student's state generates observations through multiple channels:

```python
obs = observations[10]
print("Counts:   ", obs.counts)        # e.g., [4 1 0 2]
print("Binaries: ", obs.binaries)      # e.g., [0]
print("Ordinals: ", obs.ordinals)      # e.g., [1]
```

**Interpretation:**
- `counts[0..3]` are the number of events of each engagement type (observing, participating, contributing, self-regulating). They are Poisson draws whose rate depends on the corresponding latent state.
- `binaries[0]` is whether an assessment was submitted this week.
- `ordinals[0]` is a self-reported engagement rating on a 0–3 scale.

Missing channels are `None`. During simulation, no channel is missing — that's how we build the ground truth. In real data, missing values are common.

## 2.3 Visualizing the Trajectory

Let's plot Amina's simulated semester:

```python
fig = plots.plot_latent_trajectories(
    true_states=states,
    filtered_mean=states,     # just for display: filter is Chapter 3
    filtered_std=None,
)
fig.suptitle("Amina's simulated latent state", y=1.02)
plt.savefig("amina_simulation.png", dpi=150, bbox_inches="tight")
```

You will see six subplots:
- \(e^O, e^P, e^C, e^{SR}\): the engagement dimensions, each rising and falling.
- \(P\): progress, which is a step function that jumps when assessments complete.
- \(R\): risk, which responds smoothly to engagement and progress.

## 2.4 The Discrete Status Trajectory

Let's look at the statuses over time:

```python
fig2 = plots.plot_status_probabilities(
    status_probs=np.eye(5)[statuses],   # one-hot encoding
)
fig2.suptitle("Amina's simulated status")
plt.savefig("amina_status.png", dpi=150, bbox_inches="tight")
```

Starting from `REGISTERED`, Amina transitions to `PROGRESSING` early and stays there. Occasionally the simulated trajectory might show a `STOPPED_OUT` period; in some seeds a `COMPLETED` at the end.

## 2.5 Controlling Events

By default, each event fires independently with probability 0.5. To control the event process, pass an explicit events array:

```python
rng = np.random.default_rng(1)
events = np.zeros((60, 4))

# Amina is very active in the first 20 weeks
events[:20] = (rng.random((20, 4)) < 0.7).astype(int)

# She disengages entirely for weeks 20–40
events[20:40] = 0

# She returns in weeks 40–60
events[40:] = (rng.random((20, 4)) < 0.5).astype(int)

states2, statuses2, obs2 = model.simulate_trajectory(
    n_timesteps=60, events=events, rng=rng,
)

fig3 = plots.plot_latent_trajectories(
    true_states=states2,
    filtered_mean=states2,
)
fig3.suptitle("Amina with a disengagement period", y=1.02)
plt.savefig("amina_with_gap.png", dpi=150, bbox_inches="tight")
```

You will see the engagement dimensions rise, then decay during the gap, then rise again. Risk rises during the gap and falls when she returns. This is exactly the kind of pattern static models miss — but the DLSM captures it because the latent state persists through the silence.

## 2.6 A Cohort

For a cohort simulation, use the `Simulator`:

```python
sim = Simulator(model, seed=42)
cohort = sim.simulate(n_students=50, n_timesteps=60, event_prob=0.3)

print("Cohort states:       ", cohort.states.shape)       # (50, 60, 6)
print("Cohort statuses:     ", cohort.statuses.shape)     # (50, 60)
print("Cohort observations: ", len(cohort.observations))  # 50
```

Each student gets an independent event process. Different students will show different trajectories — some engaged throughout, some disengaging, some completing early.

## 2.7 Checkpoint

You should now be able to:
- [x] Simulate a single trajectory with `simulate_trajectory`.
- [x] Simulate a cohort with `Simulator`.
- [x] Interpret the `Observation` object.
- [x] Customize the event process.
- [x] Visualize trajectories.

**Try it yourself:** Simulate a "slow starter" who does nothing for the first 20 weeks, then engages actively. Does progress stall? Does risk rise and then fall?

<details>
<summary>Answer</summary>

```python
rng = np.random.default_rng(2)
events = np.zeros((80, 4))
events[20:] = (rng.random((60, 4)) < 0.6).astype(int)

states, statuses, obs = model.simulate_trajectory(
    n_timesteps=80, events=events, rng=rng,
)
# Progress stays at 0 for 20 weeks, then rises.
# Risk rises for 20 weeks, then falls as engagement grows.
```
</details>

---

# Chapter 3 — Track a Student with the Particle Filter

**Learning objectives:**
- Run the particle filter on a trajectory.
- Interpret the filter output.
- Use ESS to diagnose filter health.
- Compute credible intervals.
- Filter a disengaging student and see the value of tracking.

---

## 3.1 The Problem

Suppose you're a learning analytics practitioner. You observe a student's LMS activity, submissions, and self-reports — that is, you observe the `Observation` objects. You do **not** observe the latent state. The particle filter's job is to infer the latent state from the observations.

## 3.2 Running the Filter

We'll reuse the trajectory from Chapter 2:

```python
sim = Simulator(model, seed=42)
states, statuses, observations = model.simulate_trajectory(
    n_timesteps=60,
    rng=np.random.default_rng(0),
)

pf = ParticleFilter(model, n_particles=1000, seed=0)
result = pf.filter(observations=observations)

print("Filtered states shape: ", result.mean_states.shape)   # (60, 6)
print("Log-likelihood:        ", round(result.log_likelihood, 2))
print("Mean ESS:              ", round(result.ess.mean(), 1))
```

The filter outputs:
- `mean_states`: posterior mean of the continuous state at each time.
- `var_states`: posterior variance.
- `status_probs`: posterior probability over the 5 statuses.
- `ess`: effective sample size at each step — a health metric.

## 3.3 Comparing Filtered and True States

Let's overlay the filter output on the ground truth:

```python
fig = plots.plot_latent_trajectories(
    true_states=states,
    filtered_mean=result.mean_states,
    filtered_std=np.sqrt(result.var_states),
)
fig.suptitle("Filter vs. truth", y=1.02)
plt.savefig("filter_vs_truth.png", dpi=150, bbox_inches="tight")
```

The black line is truth, the blue line is the filter, and the shaded band is the 95% credible interval. In good runs, the blue line tracks the black line closely, and the truth stays inside the band most of the time.

Let's quantify this:

```python
rmse = metrics.rmse(states, result.mean_states, axis=0)
print("RMSE per component:", rmse)
# Typically [0.05 0.04 0.03 0.06 0.03 0.05] or so
```

## 3.4 Credible Intervals and Coverage

The 95% credible interval should contain the truth about 95% of the time:

```python
std = np.sqrt(result.var_states)
lower = result.mean_states - 1.96 * std
upper = result.mean_states + 1.96 * std

cov = metrics.coverage(states, lower, upper)
print(f"Coverage: {cov:.1%}")   # should be close to 95%
```

If coverage is much lower than 95%, the filter is overconfident — increase process noise or particles.

## 3.5 Reading the ESS

The effective sample size (`ess`) tells you how many particles are effectively contributing to the posterior at each step:

```python
print("Min ESS:  ", result.ess.min())
print("Mean ESS: ", result.ess.mean())
print("Max ESS:  ", result.ess.max())
print("Below 30%:", (result.ess < 0.3 * pf.n_particles).mean())
```

**Rules of thumb:**

| Condition | Interpretation |
|-----------|---------------|
| Mean ESS > 0.5 N | Healthy |
| Mean ESS 0.3–0.5 N | Acceptable |
| Mean ESS < 0.3 N | Increase particles |
| ESS drops to 1 | Filter has collapsed |

## 3.6 The Disengagement Scenario

Now the payoff: filter a student through a long silence. Let's simulate the "gap" scenario from Chapter 2:

```python
rng = np.random.default_rng(3)
events = np.zeros((80, 4))
events[:30] = (rng.random((30, 4)) < 0.6).astype(int)
events[30:60] = 0
events[60:] = (rng.random((20, 4)) < 0.5).astype(int)

states_gap, statuses_gap, obs_gap = model.simulate_trajectory(
    n_timesteps=80, events=events, rng=rng,
)

pf2 = ParticleFilter(model, n_particles=1000, seed=0)
result_gap = pf2.filter(observations=obs_gap)

fig = plots.plot_latent_trajectories(
    true_states=states_gap,
    filtered_mean=result_gap.mean_states,
    filtered_std=np.sqrt(result_gap.var_states),
)
fig.suptitle("Filtering through a gap", y=1.02)
plt.savefig("filter_through_gap.png", dpi=150, bbox_inches="tight")
```

Look at what happens during weeks 30–60:
1. **Engagement decays** — the model's dynamics do this automatically, even without observations.
2. **Uncertainty grows** — the shaded band widens as time since the last observation increases.
3. **Risk rises** — the filter infers rising risk even though no observation indicated it, because the dynamics say so.

**This is the key insight.** Static models would see "no clicks" and could only output "inactive". The DLSM outputs "engagement is decaying toward zero, risk is rising, and here's our uncertainty — we'd recommend an intervention."

## 3.7 Filtering a New Student in Real Time

In deployment, you receive one observation at a time. The filter supports this:

```python
pf3 = ParticleFilter(model, n_particles=1000, seed=0)

# Process observations one at a time, simulating a real-time feed
real_time_estimates = []
for t in range(len(obs_gap)):
    partial_result = pf3.filter(obs_gap[:t+1])
    latest = partial_result.mean_states[-1]
    real_time_estimates.append(latest)

real_time_estimates = np.array(real_time_estimates)
print(real_time_estimates.shape)   # (80, 6)
```

This is inefficient (re-runs the filter each time). For production, the filter should be incremental — a feature we plan to add in a future release. For now, batch runs are the recommended approach.

## 3.8 Checkpoint

You should now be able to:
- [x] Run the particle filter on a trajectory.
- [x] Interpret ESS and diagnose filter health.
- [x] Compute RMSE and coverage.
- [x] Filter through a gap and understand why the state persists.
- [x] Produce credible intervals.

**Try it yourself:** Simulate a student who is highly active for 20 weeks, then goes silent for 40. At what week does the filter's uncertainty about engagement exceed 0.2? Does risk exceed 0.7?

<details>
<summary>Answer</summary>

```python
rng = np.random.default_rng(5)
events = np.zeros((60, 4))
events[:20] = 1                    # all events fire for 20 weeks
# weeks 20–60: silence

states, _, obs = model.simulate_trajectory(
    n_timesteps=60, events=events, rng=rng,
)
pf = ParticleFilter(model, n_particles=1000, seed=0)
r = pf.filter(obs)

std = np.sqrt(r.var_states)
# Uncertainty about eO exceeds 0.2 around week 35–40
# Risk exceeds 0.7 around week 45–50
```
</details>

---

# Chapter 4 — Estimate Parameters with PMCMC

**Learning objectives:**
- Understand what PMCMC does and why we need it.
- Run PMCMC on synthetic data and recover known parameters.
- Diagnose convergence with trace plots and R-hat.
- Tune the proposal scale.
- Know when to use PMCMC vs. fixing parameters.

---

## 4.1 The Problem

In Chapters 2–3, we used the *true* parameters to generate and filter data. But in real applications, we don't know the parameters — we need to estimate them from data. That's what PMCMC does.

The idea: given a trajectory of observations, find the posterior distribution over the 17 rate parameters (α, λ, β, γ, ω).

## 4.2 A First Estimate

Let's generate a long trajectory from known parameters and try to recover them:

```python
true_params = DLSParams.default()
true_model = DLSM(true_params)

sim = Simulator(true_model, seed=1)
data = sim.simulate(n_students=1, n_timesteps=100, event_prob=0.4)
obs = data.observations[0]
events = data.events[0]

print("Observations:", len(obs))
```

Now fit an *untouched* default model to these data. It won't recover perfectly — 100 time steps is short — but it should show convergence toward the truth:

```python
init_model = DLSM(DLSParams.default())
pmcmc = PMCMC(
    init_model,
    n_particles=500,
    n_iterations=1000,
    burn_in=200,
    thin=4,
    proposal_scale=0.02,
    seed=0,
)

result = pmcmc.fit(obs, events=events)
print("Acceptance rate:", round(result.acceptance_rate, 3))
print("Chain shape:    ", result.chain.shape)
```

## 4.3 Inspecting the Posterior

Let's visualize the posterior distributions against the true values:

```python
fig = plots.plot_posterior(
    result.chain,
    result.param_names,
    true_params=true_params.as_vector(),
)
fig.suptitle("Posterior vs. true parameters", y=1.02)
plt.savefig("posteriors.png", dpi=150, bbox_inches="tight")
```

Each subplot shows the marginal posterior for one parameter, with a black vertical line at the true value. A well-behaved run has posteriors concentrated near the true lines, with a width that reflects the information in the data.

## 4.4 A Quantified Recovery Table

Let's compute the recovery table:

```python
table = metrics.parameter_recovery_table(
    true_params=true_params.as_vector(),
    posterior_samples=result.chain,
    param_names=result.param_names,
)

for row in table:
    print(
        f"{row['parameter']:10s}  "
        f"true={row['true']:+.3f}  "
        f"post={row['posterior_mean']:+.3f}  "
        f"± {row['posterior_std']:.3f}  "
        f"covered={row['covered']}"
    )
```

Coverage is the fraction of parameters whose 95% credible interval contains the truth. With 100 time steps, expect moderate coverage — some parameters are well-identified, others less so.

## 4.5 Convergence Diagnostics

A single chain can't tell you whether it converged. Run multiple chains and compute R-hat:

```python
chains = []
for seed in (0, 1, 2):
    pmcmc = PMCMC(
        init_model, n_particles=500,
        n_iterations=1000, burn_in=200, thin=4,
        proposal_scale=0.02, seed=seed,
    )
    chains.append(pmcmc.fit(obs, events=events).chain)

stacked = np.stack(chains)      # (3, n_keep, 17)

from dlsm.inference.diagnostics import gelman_rubin, effective_sample_size

rhat = gelman_rubin(stacked)
ess  = effective_sample_size(stacked)

print("R-hat (max):", rhat.max())     # target < 1.1
print("ESS (min):  ", ess.min())      # target > 500
```

If R-hat exceeds 1.1, the chains haven't mixed — increase iterations or tune the proposal.

## 4.6 Tuning the Proposal Scale

The proposal scale controls how far each MCMC step proposes to move. Too small → slow mixing; too large → almost everything rejected. Target 20–40% acceptance.

```python
for scale in [0.005, 0.02, 0.05, 0.1]:
    pmcmc = PMCMC(
        init_model, n_particles=300,
        n_iterations=500, burn_in=100, thin=2,
        proposal_scale=scale, seed=0,
    )
    r = pmcmc.fit(obs, events=events)
    print(f"scale={scale:.3f}  accept={r.acceptance_rate:.2%}")
```

Pick the scale whose acceptance rate is nearest 0.3.

## 4.7 The Cost of Accuracy

PMCMC is expensive. Every MCMC iteration runs a full particle filter. Rough cost:

- \(N_{\text{particles}} \times T\) operations per iteration.
- 10,000 iterations with \(N_p = 2000, T = 100\) is on the order of hours.

For exploration, use coarse settings; for the final run, use fine settings. Here's a template:

```python
# Exploratory run — fast
pmcmc_fast = PMCMC(
    init_model, n_particles=200, n_iterations=500,
    burn_in=100, thin=2, proposal_scale=0.02, seed=0,
)
r_fast = pmcmc_fast.fit(obs, events=events)
print("Fast acceptance:", r_fast.acceptance_rate)

# Final run — slow but accurate
pmcmc_final = PMCMC(
    init_model, n_particles=2000, n_iterations=10000,
    burn_in=2000, thin=10, proposal_scale=0.02, seed=0,
)
# r_final = pmcmc_final.fit(obs, events=events)  # uncomment to run
```

## 4.8 Checkpoint

You should now be able to:
- [x] Run PMCMC on a synthetic trajectory.
- [x] Interpret the posterior and recovery table.
- [x] Diagnose convergence with R-hat and ESS.
- [x] Tune the proposal scale to target ~30% acceptance.
- [x] Balance accuracy against computational cost.

**Try it yourself:** Generate three trajectories with different true parameters (e.g., high α, low λ vs. low α, high λ). Fit each with PMCMC. Which parameter sets are easier to recover? Why?

<details>
<summary>Hint</summary>

Fast dynamics (high α, high λ) produce more observable transitions per unit time and are generally easier to identify than slow dynamics. Also, longer trajectories give more information. Try `n_timesteps=200` and see whether coverage improves.
</details>

---

# Chapter 5 — Apply to a Cohort and Evaluate

**Learning objectives:**
- Filter and evaluate an entire cohort.
- Compute prediction accuracy for risk, engagement, and progress.
- Compare against baselines.
- Produce a cohort-level summary.

---

## 5.1 Simulating a Realistic Cohort

Let's build a mixed cohort with three archetypes:

```python
rng = np.random.default_rng(42)
n_students = 60
n_timesteps = 60

events = np.zeros((n_students, n_timesteps, 4))
archetype = rng.integers(0, 3, size=n_students)

for i in range(n_students):
    if archetype[i] == 0:   # "steady" — consistent engagement
        events[i] = (rng.random((n_timesteps, 4)) < 0.4).astype(int)
    elif archetype[i] == 1: # "front-loaded" — engaged early, then fades
        events[i, :30] = (rng.random((30, 4)) < 0.6).astype(int)
        events[i, 30:] = (rng.random((30, 4)) < 0.1).astype(int)
    else:                    # "late bloomer" — disengaged early, then engaged
        events[i, :30] = (rng.random((30, 4)) < 0.1).astype(int)
        events[i, 30:] = (rng.random((30, 4)) < 0.6).astype(int)

# Simulate each archetype
cohort_states = np.zeros((n_students, n_timesteps, 6))
cohort_statuses = np.zeros((n_students, n_timesteps), dtype=int)
cohort_obs = []

for i in range(n_students):
    s, st, ob = model.simulate_trajectory(
        n_timesteps=n_timesteps, events=events[i], rng=rng,
    )
    cohort_states[i] = s
    cohort_statuses[i] = st
    cohort_obs.append(ob)

print("Cohort simulated:", cohort_states.shape)
```

## 5.2 Batch Filtering

```python
from joblib import Parallel, delayed

def filter_one(i):
    pf = ParticleFilter(model, n_particles=500, seed=1000 + i)
    return pf.filter(cohort_obs[i], events=events[i])

results = Parallel(n_jobs=-1)(
    delayed(filter_one)(i) for i in range(n_students)
)
print("Filtered", len(results), "students")
```

Or sequentially if you prefer:

```python
results = [
    ParticleFilter(model, n_particles=500, seed=1000+i).filter(
        cohort_obs[i], events=events[i],
    )
    for i in range(n_students)
]
```

## 5.3 Cohort-Level RMSE

```python
true_eng = cohort_states[..., :4]          # (n_students, T, 4)
est_eng = np.stack([r.mean_states[:, :4] for r in results])

rmse_per_student = np.sqrt(
    ((true_eng - est_eng) ** 2).mean(axis=(1, 2))
)
print(f"Cohort RMSE: {rmse_per_student.mean():.4f} "
      f"± {rmse_per_student.std():.4f}")
```

## 5.4 Risk Prediction

Suppose the goal is to predict, at each time step, whether the student will withdraw within the next 5 steps:

```python
def withdrawal_within(statuses, t, horizon=5):
    """1 if status becomes WITHDRAWN within [t, t+horizon)"""
    window = statuses[t:t+horizon]
    return int((window == Status.WITHDRAWN).any())

# Build labels and predicted probabilities
y_true = []
y_score = []
for i, r in enumerate(results):
    for t in range(n_timesteps - 5):
        label = withdrawal_within(cohort_statuses[i], t, horizon=5)
        # Use posterior probability of WITHDRAWN
        prob = r.status_probs[t, Status.WITHDRAWN]
        y_true.append(label)
        y_score.append(prob)

y_true = np.array(y_true)
y_score = np.array(y_score)

from dlsm.evaluation.metrics import auc, brier_score
print("AUC:         ", round(auc(y_score, y_true), 3))
print("Brier score: ", round(brier_score(y_score, y_true), 3))
```

AUC above 0.85 and Brier below 0.15 indicates good prediction.

## 5.5 Calibration

Calibration tells you whether predicted probabilities match observed frequencies:

```python
fig = plots.plot_calibration(
    predicted_prob=y_score,
    outcomes=y_true,
    n_bins=10,
)
fig.suptitle("Risk prediction calibration", y=1.02)
plt.savefig("calibration.png", dpi=150, bbox_inches="tight")
```

Well-calibrated predictions fall near the diagonal. Systematic deviation (e.g., always under-predicting risk) indicates the model needs retuning.

## 5.6 Comparing Against Baselines

Let's compare DLSM against a static logistic regression baseline:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Static features: mean counts per student
static_features = np.stack([
    np.stack([obs.counts for obs in cohort_obs[i]]).mean(axis=0)
    for i in range(n_students)
])
print("Static features shape:", static_features.shape)   # (n, 4)

# Labels: did the student withdraw at all?
final_status = cohort_statuses[:, -1]
y_binary = (final_status == Status.WITHDRAWN).astype(int)

clf = LogisticRegression(max_iter=1000)
clf.fit(static_features, y_binary)
static_probs = clf.predict_proba(static_features)[:, 1]

print("Static AUC:  ", round(roc_auc_score(y_binary, static_probs), 3))
print("DLSM AUC:    ", round(auc(y_score, y_true), 3))
```

The DLSM should outperform the static baseline because it captures temporal dynamics and produces calibrated, per-step risk estimates rather than a single cohort-level prediction.

## 5.7 Cohort Summary

Produce a clean summary table:

```python
import pandas as pd

summary = []
for i, r in enumerate(results):
    summary.append({
        "student": i,
        "archetype": int(archetype[i]),
        "final_engagement": float(r.mean_states[-1, :4].mean()),
        "final_progress": float(r.mean_states[-1, 4]),
        "final_risk": float(r.mean_states[-1, 5]),
        "final_status": int(r.mean_status[-1]),
        "max_risk": float(r.mean_states[:, 5].max()),
    })
df = pd.DataFrame(summary)
print(df.groupby("archetype")[["final_engagement", "final_risk"]].mean())
```

The "front-loaded" archetype should show lower final engagement and higher final risk; the "late bloomer" should show the reverse. This validates that the filter captures the underlying dynamics.

## 5.8 Checkpoint

You should now be able to:
- [x] Simulate a heterogeneous cohort.
- [x] Batch-filter with parallelism.
- [x] Compute cohort-level RMSE.
- [x] Predict withdrawal with AUC and Brier score.
- [x] Compare against a logistic regression baseline.
- [x] Produce a cohort summary table.

**Try it yourself:** Change the horizon from 5 to 10 steps. How does AUC change? A shorter horizon should give higher AUC (easier prediction) but less lead time.

---

# Chapter 6 — Extend the Model for Your Own Data

**Learning objectives:**
- Map real LMS data to the DLSM's observation model.
- Customize observation weight matrices.
- Handle missing data.
- Write a custom drift function.
- Deploy the model to a new student.

---

## 6.1 Mapping Your Data

Suppose you have a CSV of daily LMS activity:

```
student_id, date, page_views, video_plays, forum_reads, forum_posts,
           assignments_submitted, self_report
```

The DLSM needs six channels. Two common designs:

**Design A — One channel per engagement dimension.**

| Channel | Type | Loads on | How to compute |
|---------|------|----------|----------------|
| counts[0] | Poisson | \(e^O\) | `page_views + video_plays` |
| counts[1] | Poisson | \(e^P\) | `forum_reads` |
| counts[2] | Poisson | \(e^C\) | `forum_posts` |
| counts[3] | Poisson | \(e^{SR}\) | `self_regulated_actions` |
| binaries[0] | Bernoulli | \(P\) | `assignments_submitted > 0` |
| ordinals[0] | Ordinal | \(e^{SR}\) | `self_report` |

**Design B — Multiple channels per dimension.**

If you have several distinct observing signals (views, plays, downloads), give each its own channel that loads on \(e^O\) with different weights. This lets the model learn that views are a stronger signal than downloads.

## 6.2 Building Observations from a DataFrame

```python
import pandas as pd
from dlsm.core.observations import Observation

def df_to_observations(df: pd.DataFrame, freq="W") -> list[Observation]:
    """
    df: columns student_id, date, page_views, video_plays, forum_reads,
        forum_posts, assignments_submitted, self_report.
    Returns a list of Observation objects, one per time step.
    """
    df = df.set_index("date").sort_index()

    # Resample to weekly windows
    agg = df.resample(freq).agg({
        "page_views": "sum",
        "video_plays": "sum",
        "forum_reads": "sum",
        "forum_posts": "sum",
        "assignments_submitted": "sum",
        "self_report": "mean",
    }).fillna(0)

    obs_list = []
    for _, row in agg.iterrows():
        # Counts: [observing, participating, contributing, self-regulated]
        counts = np.array([
            int(row["page_views"] + row["video_plays"]),
            int(row["forum_reads"]),
            int(row["forum_posts"]),
            0,   # self-regulated proxy: leave at 0 for now
        ])
        binaries = np.array([int(row["assignments_submitted"] > 0)])
        ordinals = np.array([int(row["self_report"])]) \
                    if row["self_report"] > 0 else None
        obs_list.append(Observation(
            counts=counts,
            binaries=binaries,
            ordinals=ordinals,
        ))
    return obs_list
```

## 6.3 Customizing the Observation Weights

The default weights assume each count channel maps one-to-one to a specific engagement dimension. If your channels are correlated or load differently, override the weights:

```python
params = DLSParams.default()

# Row 0: page_views -> eO
# Row 1: forum_reads -> eO and eP (participation requires observing)
# Row 2: forum_posts -> eC
# Row 3: self-regulation proxy -> eSR
params.w_count = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.3, 0.7, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
])

# Binaries: submission depends on progress
params.w_binary = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

# Ordinals: self-report depends on self-regulated engagement
params.w_ordinal = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
params.ordinal_thresholds = np.array([-0.5, 0.5, 1.5])
```

Now filtering a new student uses these weights automatically.

## 6.4 Handling Missing Data

Real data is often missing. The DLSM handles missing channels gracefully — just pass `None`:

```python
# A week with only LMS activity (no submission, no self-report)
obs = Observation(
    counts=np.array([12, 3, 0, 0]),
    binaries=None,      # assignment data missing
    ordinals=None,      # self-report missing
)

# A week with no data at all (student inactive, no logs)
obs_empty = Observation(
    counts=np.array([0, 0, 0, 0]),
    binaries=None,
    ordinals=None,
)
```

Both are handled by the filter. The second is *informative* — the student really was inactive — while the first is truly missing on some channels.

If an entire week is missing from your data (e.g., a system outage), the safest approach is to interpolate the counts as zeros and mark ordinals as `None`. Zeros during a genuine outage will bias the engagement estimate slightly downward; if outages are common, consider adding a "missing" indicator channel.

## 6.5 Writing a Custom Drift

Suppose you believe engagement dynamics differ by course level (e.g., 100-level vs. 400-level courses). Add a course-level offset:

```python
from dlsm.core.dynamics import Dynamics

class CourseLevelDynamics(Dynamics):
    """Course-level random effect on engagement drift."""

    def __init__(self, params, course_offset=0.0):
        super().__init__(params)
        self.course_offset = course_offset

    def drift(self, x, events, dt=None):
        base = super().drift(x, events, dt=dt)
        # Shift all four engagement drift components
        base[:, :4] += self.course_offset
        return base

# Use it
model_custom = DLSM(params)
model_custom.dynamics = CourseLevelDynamics(params, course_offset=0.02)
```

Now the drift is slightly higher for this course, modeling the fact that students in this course are, on average, more engaged.

## 6.6 Deploying to a New Student

The deployment workflow:

1. **Fit the model offline** on historical data using PMCMC.
2. **Extract the posterior mean** parameters.
3. **For each new student, filter their observations** as they arrive.

```python
# Step 1: Fit offline (assume we have historical data)
historical_obs = cohort_obs[0]
historical_events = events[0]

pmcmc = PMCMC(model, n_particles=1000, n_iterations=2000,
              burn_in=500, thin=5, seed=0)
fit_result = pmcmc.fit(historical_obs, events=historical_events)

# Step 2: Extract posterior mean
theta_hat = fit_result.posterior_mean()
params_hat = DLSParams.from_vector(theta_hat, template=params)
model_hat = DLSM(params_hat)

# Step 3: Filter a new student in real time
new_student_obs = cohort_obs[5]      # a student we've never seen
pf_new = ParticleFilter(model_hat, n_particles=1000, seed=0)
new_result = pf_new.filter(new_student_obs, events=events[5])

# Extract actionable signals
latest_risk = new_result.mean_states[-1, 5]
latest_risk_std = np.sqrt(new_result.var_states[-1, 5])

# Trigger an intervention if risk is high and certain
if latest_risk > 0.7 and latest_risk_std < 0.15:
    print(f"INTERVENTION RECOMMENDED: risk = {latest_risk:.2f}")
```

## 6.7 A Full Pipeline

Here's a reusable function that ties everything together:

```python
def analyze_student(
    df: pd.DataFrame,
    fitted_model: DLSM,
    n_particles: int = 1000,
    seed: int = 0,
) -> dict:
    """
    Run the full pipeline on a single student's raw data.

    Returns a dict with filtered states, statuses, and risk alerts.
    """
    obs = df_to_observations(df)
    pf = ParticleFilter(fitted_model, n_particles=n_particles, seed=seed)
    result = pf.filter(obs)

    risk = result.mean_states[:, 5]
    risk_std = np.sqrt(result.var_states[:, 5])

    alerts = [
        {"t": t, "risk": float(risk[t]), "std": float(risk_std[t])}
        for t in range(len(risk))
        if risk[t] > 0.7 and risk_std[t] < 0.15
    ]

    return {
        "mean_states": result.mean_states,
        "status_probs": result.status_probs,
        "log_likelihood": result.log_likelihood,
        "alerts": alerts,
    }

# Usage
# result = analyze_student(student_df, model_hat)
# print(result["alerts"])
```

## 6.8 Checkpoint

You should now be able to:
- [x] Map your data to the observation model.
- [x] Build `Observation` objects from a DataFrame.
- [x] Customize observation weights for your context.
- [x] Handle missing data gracefully.
- [x] Write a custom drift function.
- [x] Run the complete fit-then-deploy pipeline.

**Try it yourself:** Take any time series dataset you have (LMS logs, e-commerce visits, fitness trackers) and map it to the DLSM's six-channel layout. Simulate a few students. Does the filter produce sensible trajectories?

---

# Where to Go Next

You've completed the tutorial. Here are some directions:

## For practitioners

- **Deploy the model** behind an early-warning dashboard.
- **Set up a nightly job** that filters each active student and triggers alerts.
- **Integrate with your LMS** by writing an adapter that reads logs and calls `analyze_student`.

## For researchers

- **Run the paper's simulation study** — see `examples/parameter_recovery.py`.
- **Fit to your own institutional data** — the empirical validation protocol is in Section 6 of the manuscript.
- **Compare against baselines** — logistic regression, random forest, DBN. The `metrics` module has AUC, Brier, RMSE, and coverage built in.
- **Extend the model** — add emotional engagement, hierarchical random effects, or POMDP-based interventions.

## For developers

- **Read the tests** — `tests/test_core.py` shows the API in detail.
- **Add new channels** — subclass `ObservationModel`.
- **Contribute** — see the contributing guide in the README.

## Reference material

| Resource | Where |
|----------|-------|
| User guide | `USER_GUIDE.md` |
| API cheat sheet | Section 16 of the user guide |
| Conceptual foundations | The manuscript |
| Runnable examples | `examples/` |
| Tests | `tests/` |
| Issue tracker | GitHub |

## A final thought

The DLSM is fundamentally about **taking latent states seriously**. Engagement is not what you count; it's what the counts are trying to tell you about. When you use this library, you're making a theoretical commitment: that Amina's engagement persists through her silence, that risk rises before withdrawal, and that a well-designed measurement model can infer what isn't directly observed.

Use it well.

---

**Last updated:** 2026-10-02 · **Version:** 0.1.0 · **Feedback:** [open an issue](https://github.com/hongxueharriswang/dlsm-lib/issues)
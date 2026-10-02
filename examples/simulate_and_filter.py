"""Example: simulate a cohort and run the particle filter."""

import numpy as np
import matplotlib.pyplot as plt

from dlsm import DLSM, DLSParams, ParticleFilter, Simulator, plots


def main():
    # 1. Model
    params = DLSParams.default()
    model = DLSM(params)

    # 2. Simulate
    sim = Simulator(model, seed=42)
    data = sim.simulate(n_students=20, n_timesteps=60, event_prob=0.3)

    # 3. Filter first student
    pf = ParticleFilter(model, n_particles=1000, seed=0)
    result = pf.filter(
        data.observations[0],
        events=data.events[0],
    )

    print(f"Log-likelihood: {result.log_likelihood:.2f}")
    print(f"Mean ESS: {result.ess.mean():.1f}")

    # 4. Plot
    fig = plots.plot_latent_trajectories(
        true_states=data.states[0],
        filtered_mean=result.mean_states,
        filtered_std=np.sqrt(result.var_states),
    )
    fig.savefig("trajectories.png", dpi=150)

    fig2 = plots.plot_status_probabilities(
        result.status_probs,
        true_status=data.statuses[0],
    )
    fig2.savefig("status_probs.png", dpi=150)

    print("Saved trajectories.png and status_probs.png")


if __name__ == "__main__":
    main()
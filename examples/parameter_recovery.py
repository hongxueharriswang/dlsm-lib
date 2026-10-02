"""Example: parameter recovery with PMCMC."""

import numpy as np
import matplotlib.pyplot as plt

from dlsm import DLSM, DLSParams, Simulator, PMCMC, plots
from dlsm.evaluation import metrics


def main():
    # 1. True parameters
    true_params = DLSParams.default()
    true_model = DLSM(true_params)

    # 2. Simulate one long trajectory
    sim = Simulator(true_model, seed=1)
    data = sim.simulate(n_students=1, n_timesteps=80, event_prob=0.4)

    # 3. Estimate parameters
    init_model = DLSM(DLSParams.default())
    pmcmc = PMCMC(
        init_model,
        n_particles=500,
        n_iterations=2000,
        burn_in=500,
        thin=5,
        proposal_scale=0.02,
        seed=0,
    )
    result = pmcmc.fit(data.observations[0], events=data.events[0])

    print(f"Acceptance rate: {result.acceptance_rate:.2%}")
    print(f"Posterior mean: {result.posterior_mean()}")

    # 4. Recovery table
    table = metrics.parameter_recovery_table(
        true_params.as_vector(),
        result.chain,
        result.param_names,
    )
    for row in table:
        print(f"{row['parameter']:10s}  true={row['true']:+.3f}  "
              f"post={row['posterior_mean']:+.3f}  "
              f"covered={row['covered']}")

    # 5. Plot posteriors
    fig = plots.plot_posterior(
        result.chain,
        result.param_names,
        true_params=true_params.as_vector(),
    )
    fig.savefig("posteriors.png", dpi=150)
    print("Saved posteriors.png")


if __name__ == "__main__":
    main()
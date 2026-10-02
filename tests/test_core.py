"""Test suite for the DLSM core module.

Covers:
- dlsm._numerics        (sigmoid, logsumexp, normalize_log_weights)
- dlsm.core.parameters  (DLSParams)
- dlsm.core.state       (Status, DLSMState)
- dlsm.core.constraints (project_ordered_cone)
- dlsm.core.dynamics    (Dynamics)
- dlsm.core.observations(ObservationModel, Observation)
"""

from __future__ import annotations

import numpy as np
import pytest

from dlsm._numerics import (
    sigmoid,
    log_sigmoid,
    logsumexp,
    normalize_log_weights,
)
from dlsm.core.parameters import (
    DLSParams,
    STATUS_LABELS,
    N_STATUS,
    N_ENGAGEMENT,
    N_CONTINUOUS,
    _default_status_coefficients,
)
from dlsm.core.state import DLSMState, Status
from dlsm.core.constraints import project_ordered_cone, _pava_decreasing
from dlsm.core.dynamics import Dynamics
from dlsm.core.observations import ObservationModel, Observation


# ======================================================================
# Fixtures
# ======================================================================
@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def params():
    return DLSParams.default()


@pytest.fixture
def dynamics(params):
    return Dynamics(params)


@pytest.fixture
def obs_model(params):
    return ObservationModel(params)


# ======================================================================
# _numerics
# ======================================================================
class TestNumerics:

    def test_sigmoid_basic_values(self):
        assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
        assert sigmoid(np.array([100.0]))[0] == pytest.approx(1.0)
        assert sigmoid(np.array([-100.0]))[0] == pytest.approx(0.0)

    def test_sigmoid_no_overflow(self):
        # Very large / very small values should not produce NaN or inf
        x = np.array([-1e6, -1000.0, 0.0, 1000.0, 1e6])
        y = sigmoid(x)
        assert np.all(np.isfinite(y))
        assert np.all(y >= 0.0) and np.all(y <= 1.0)

    def test_sigmoid_symmetry(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert np.allclose(sigmoid(x) + sigmoid(-x), 1.0)

    def test_log_sigmoid_matches_log_of_sigmoid(self):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        assert np.allclose(log_sigmoid(x), np.log(sigmoid(x)))

    def test_log_sigmoid_stable_for_large_negative(self):
        x = np.array([-1000.0])
        assert np.isfinite(log_sigmoid(x)[0])
        # log(sigmoid(-1000)) ~= -1000
        assert log_sigmoid(x)[0] == pytest.approx(-1000.0, abs=1e-9)

    def test_logsumexp_scalar(self):
        assert logsumexp(np.array([1.0, 2.0, 3.0])) == pytest.approx(
            np.log(np.sum(np.exp([1.0, 2.0, 3.0])))
        )

    def test_logsumexp_axis(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = logsumexp(a, axis=1)
        expected = np.log(np.sum(np.exp(a), axis=1))
        assert np.allclose(out, expected)

    def test_logsumexp_stable(self):
        a = np.array([1000.0, 1000.0, 1000.0])
        # Without stabilization this would overflow
        assert np.isfinite(logsumexp(a))
        assert logsumexp(a) == pytest.approx(1000.0 + np.log(3.0))

    def test_normalize_log_weights_sums_to_one(self):
        log_w = np.log(np.array([0.1, 0.2, 0.7]))
        w = normalize_log_weights(log_w)
        assert w.sum() == pytest.approx(1.0)

    def test_normalize_log_weights_uniform_on_degenerate(self):
        log_w = np.full(4, -np.inf)
        w = normalize_log_weights(log_w)
        assert np.allclose(w, 0.25)

    def test_normalize_log_weights_preserves_order(self):
        log_w = np.array([-10.0, 0.0, -5.0])
        w = normalize_log_weights(log_w)
        assert w[1] > w[2] > w[0]


# ======================================================================
# parameters
# ======================================================================
class TestDLSParams:

    def test_default_constructs(self):
        p = DLSParams.default()
        assert p.alpha_O > 0
        assert p.lambda_O > 0
        assert p.dt == 1.0

    def test_shapes(self, params):
        assert params.q_diag.shape == (N_CONTINUOUS,)
        assert params.w_count.shape[1] == N_CONTINUOUS
        assert params.w_binary.shape[1] == N_CONTINUOUS
        assert params.w_ordinal.shape[1] == N_CONTINUOUS
        assert params.w_status.shape == (
            N_STATUS, N_STATUS, params.n_status_features
        )

    def test_default_w_count_maps_engagement_dims(self, params):
        # The default should map each of the four engagement dims
        # to a separate Poisson channel
        eye = np.eye(N_ENGAGEMENT)
        assert np.allclose(params.w_count[:, :N_ENGAGEMENT], eye)
        assert np.allclose(params.w_count[:, N_ENGAGEMENT:], 0.0)

    def test_default_absorbing_states(self):
        w = _default_status_coefficients()
        # Withdrawn (3) and Completed (4) should have huge negative logits
        # for any outgoing transition
        for s in (3, 4):
            for s2 in range(N_STATUS):
                if s2 == s:
                    continue
                assert np.all(w[s, s2, :] <= -5.0)

    def test_invalid_q_diag_raises(self):
        with pytest.raises(ValueError, match="q_diag"):
            DLSParams(q_diag=np.array([0.1, 0.1]))   # too short

    def test_invalid_w_status_shape_raises(self):
        with pytest.raises(ValueError, match="w_status"):
            DLSParams(w_status=np.zeros((2, 2, 3)))

    def test_as_vector_from_vector_roundtrip(self):
        p = DLSParams.default()
        v = p.as_vector()
        p2 = DLSParams.from_vector(v, template=p)
        # Only the rate parameters are serialized; the rest come from template
        assert v.shape == (17,)
        assert p2.alpha_O == pytest.approx(p.alpha_O)
        assert p2.omega_3 == pytest.approx(p.omega_3)
        # Non-serialized fields should be preserved from the template
        assert np.allclose(p2.q_diag, p.q_diag)
        assert np.allclose(p2.w_status, p.w_status)

    def test_as_vector_length_matches_param_names(self):
        from dlsm.inference.pmcmc import PMCMC
        assert DLSParams.default().as_vector().shape[0] == len(PMCMC.PARAM_NAMES)

    def test_custom_assessment_weights_are_float(self):
        p = DLSParams(assessment_weights=[1, 2, 3])
        assert p.assessment_weights.dtype == np.float64

    def test_status_labels_consistent(self):
        assert len(STATUS_LABELS) == N_STATUS


# ======================================================================
# state
# ======================================================================
class TestState:

    def test_status_enum_values(self):
        assert int(Status.REGISTERED) == 0
        assert int(Status.PROGRESSING) == 1
        assert int(Status.STOPPED_OUT) == 2
        assert int(Status.WITHDRAWN) == 3
        assert int(Status.COMPLETED) == 4

    def test_construct_valid(self):
        s = DLSMState(
            status=Status.PROGRESSING,
            continuous=np.array([0.5, 0.4, 0.2, 0.6, 0.3, 0.1]),
        )
        assert s.status == 1
        assert s.engagement.shape == (N_ENGAGEMENT,)
        assert s.e_O == pytest.approx(0.5)
        assert s.e_P == pytest.approx(0.4)
        assert s.e_C == pytest.approx(0.2)
        assert s.e_SR == pytest.approx(0.6)
        assert s.progress == pytest.approx(0.3)
        assert s.risk == pytest.approx(0.1)

    def test_construct_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            DLSMState(status=0, continuous=np.array([0.1, 0.2]))

    def test_construct_invalid_status_raises(self):
        with pytest.raises(ValueError, match="status"):
            DLSMState(status=99, continuous=np.zeros(N_CONTINUOUS))

    def test_continuous_cast_to_float(self):
        s = DLSMState(status=0, continuous=[0, 0, 0, 0, 0, 0])
        assert s.continuous.dtype == np.float64


# ======================================================================
# constraints
# ======================================================================
class TestConstraints:

    def test_projection_satisfies_ordering(self):
        e = np.array([[0.5, 0.7, 0.3, 0.2]])
        p = project_ordered_cone(e)
        assert p[0, 0] >= p[0, 1] >= p[0, 2]

    def test_projection_clips_to_unit_interval(self):
        e = np.array([[-0.5, 0.5, 1.5, 2.0]])
        p = project_ordered_cone(e)
        assert np.all(p >= 0.0)
        assert np.all(p <= 1.0)

    def test_projection_preserves_fourth_component(self):
        e = np.array([[0.5, 0.5, 0.5, 0.3]])
        p = project_ordered_cone(e)
        assert p[0, 3] == pytest.approx(0.3)

    def test_projection_idempotent(self):
        e = np.array([[0.8, 0.5, 0.2, 0.9], [0.1, 0.9, 0.3, 0.4]])
        p1 = project_ordered_cone(e)
        p2 = project_ordered_cone(p1)
        assert np.allclose(p1, p2)

    def test_projection_batch(self):
        e = np.random.default_rng(0).random((50, 4))
        p = project_ordered_cone(e)
        # All rows must satisfy the constraint
        assert np.all(p[:, 0] >= p[:, 1] - 1e-9)
        assert np.all(p[:, 1] >= p[:, 2] - 1e-9)
        assert np.all(p >= 0.0) and np.all(p <= 1.0)

    def test_projection_preserves_shape(self):
        for shape in [(4,), (10, 4), (3, 5, 4)]:
            e = np.random.default_rng(1).random(shape)
            p = project_ordered_cone(e)
            assert p.shape == shape

    def test_projection_already_ordered_no_change(self):
        e = np.array([[0.9, 0.5, 0.2, 0.7]])
        p = project_ordered_cone(e)
        assert np.allclose(p, e)

    def test_pava_decreasing_simple(self):
        y = np.array([0.2, 0.8, 0.5])
        out = _pava_decreasing(y)
        assert out[0] >= out[1] >= out[2]

    def test_pava_decreasing_already_ordered(self):
        y = np.array([0.9, 0.5, 0.1])
        out = _pava_decreasing(y)
        assert np.allclose(out, y)

    def test_pava_decreasing_all_violated(self):
        y = np.array([0.1, 0.5, 0.9])
        out = _pava_decreasing(y)
        # Should be constant equal to mean
        assert np.allclose(out, y.mean())

    def test_pava_preserves_sum(self):
        y = np.array([0.3, 0.9, 0.1, 0.7, 0.2])
        out = _pava_decreasing(y)
        assert out.sum() == pytest.approx(y.sum())

    def test_pava_single_element(self):
        y = np.array([0.5])
        assert _pava_decreasing(y)[0] == pytest.approx(0.5)


# ======================================================================
# dynamics
# ======================================================================
class TestDynamics:

    # ---- drift -----------------------------------------------------------
    def test_drift_shape(self, dynamics):
        x = np.tile(np.array([0.3, 0.2, 0.1, 0.4, 0.5, 0.6]), (10, 1))
        ev = np.tile(np.array([1, 1, 1, 1]), (10, 1))
        d = dynamics.drift(x, ev)
        assert d.shape == (10, N_CONTINUOUS)

    def test_drift_no_events_decreases_engagement(self, dynamics):
        x = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
        ev = np.zeros((1, 4))
        d = dynamics.drift(x, ev)
        # Decay terms dominate; engagement should decrease
        assert d[0, 0] < 0
        assert d[0, 1] < 0
        assert d[0, 2] < 0
        assert d[0, 3] < 0

    def test_drift_observing_increases_with_access_event(self, dynamics):
        x = np.array([[0.1, 0.1, 0.1, 0.1, 0.5, 0.5]])
        ev = np.array([[1, 0, 0, 0]])
        d = dynamics.drift(x, ev)
        # With access, e^O should grow
        assert d[0, 0] > 0

    def test_drift_zero_time_step(self, dynamics):
        x = np.tile(np.array([0.3, 0.2, 0.1, 0.4, 0.5, 0.6]), (5, 1))
        ev = np.ones((5, 4))
        d = dynamics.drift(x, ev, dt=0.0)
        assert np.allclose(d, 0.0)

    # ---- step ------------------------------------------------------------
    def test_step_shape(self, dynamics):
        x = np.tile(np.array([0.3, 0.2, 0.1, 0.4, 0.5, 0.6]), (7, 1))
        ev = np.ones((7, 4))
        x_next = dynamics.step(x, ev, rng=np.random.default_rng(0))
        assert x_next.shape == (7, N_CONTINUOUS)

    def test_step_respects_bounds(self, dynamics, rng):
        x = rng.random((100, N_CONTINUOUS))
        ev = rng.integers(0, 2, size=(100, 4))
        x_next = dynamics.step(x, ev, rng=rng)
        assert np.all(x_next >= 0.0) and np.all(x_next <= 1.0)

    def test_step_respects_ordered_cone(self, dynamics, rng):
        x = rng.random((100, N_CONTINUOUS))
        ev = rng.integers(0, 2, size=(100, 4))
        x_next = dynamics.step(x, ev, rng=rng)
        assert np.all(x_next[:, 0] >= x_next[:, 1] - 1e-9)
        assert np.all(x_next[:, 1] >= x_next[:, 2] - 1e-9)

    def test_step_no_noise_is_deterministic(self, dynamics):
        x = np.tile(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), (3, 1))
        ev = np.ones((3, 4))
        a = dynamics.step(x, ev, add_noise=False)
        b = dynamics.step(x, ev, add_noise=False)
        assert np.allclose(a, b)

    def test_step_with_noise_is_stochastic(self, dynamics, rng):
        x = np.tile(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), (3, 1))
        ev = np.ones((3, 4))
        a = dynamics.step(x, ev, rng=rng, add_noise=True)
        b = dynamics.step(x, ev, rng=rng, add_noise=True)
        assert not np.allclose(a, b)

    def test_step_progress_jump_increases_progress(self, dynamics):
        x = np.tile(np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.5]), (1, 1))
        ev = np.ones((1, 4))
        x_plain = dynamics.step(x, ev, add_noise=False)
        x_jump = dynamics.step(x, ev, assessment_score=1.0, add_noise=False)
        assert x_jump[0, 4] > x_plain[0, 4]

    # ---- progress jump ---------------------------------------------------
    def test_progress_jump_shape(self, dynamics):
        x = np.tile(np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.5]), (4, 1))
        jump = dynamics.progress_jump(x, score=0.8)
        assert jump.shape == (4,)

    def test_progress_jump_bounded(self, dynamics):
        x = np.tile(np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.5]), (10, 1))
        jump = dynamics.progress_jump(x, score=1.0)
        assert np.all(jump > 0)
        assert np.all(jump <= dynamics.params.assessment_weights.max())

    def test_progress_jump_monotone_in_score(self, dynamics):
        x = np.tile(np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.5]), (1, 1))
        j_lo = dynamics.progress_jump(x, score=0.0)[0]
        j_hi = dynamics.progress_jump(x, score=1.0)[0]
        assert j_hi > j_lo

    # ---- status transitions ---------------------------------------------
    def test_status_logits_shape(self, dynamics):
        x = np.tile(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), (3, 1))
        s = np.array([0, 1, 2])
        logits = dynamics.status_logits(x, s)
        assert logits.shape == (3, N_STATUS)

    def test_status_probs_sum_to_one(self, dynamics, rng):
        x = rng.random((50, N_CONTINUOUS))
        s = rng.integers(0, N_STATUS, size=50)
        probs = dynamics.status_transition_probs(x, s)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_status_probs_nonnegative(self, dynamics, rng):
        x = rng.random((50, N_CONTINUOUS))
        s = rng.integers(0, N_STATUS, size=50)
        probs = dynamics.status_transition_probs(x, s)
        assert np.all(probs >= 0.0)

    def test_absorbing_states_absorbing(self, dynamics):
        # Withdrawn (3) and Completed (4) should almost surely stay put
        x = np.tile(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), (2, 1))
        s = np.array([3, 4])
        probs = dynamics.status_transition_probs(x, s)
        assert probs[0, 3] > 0.99
        assert probs[1, 4] > 0.99

    def test_sample_status_shape(self, dynamics, rng):
        x = rng.random((20, N_CONTINUOUS))
        s = rng.integers(0, N_STATUS, size=20)
        s_next = dynamics.sample_status(x, s, rng=rng)
        assert s_next.shape == (20,)
        assert np.all(s_next >= 0) and np.all(s_next < N_STATUS)

    def test_sample_status_absorbing_deterministic(self, dynamics, rng):
        x = np.tile(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), (100, 1))
        s = np.full(100, 3)   # all Withdrawn
        s_next = dynamics.sample_status(x, s, rng=rng)
        # Almost all should remain Withdrawn
        assert (s_next == 3).mean() > 0.9


# ======================================================================
# observations
# ======================================================================
class TestObservationModel:

    # ---- Poisson ---------------------------------------------------------
    def test_poisson_ll_shape(self, obs_model):
        x = np.tile(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), (10, 1))
        y = np.ones((10, 4), dtype=int)
        ll = obs_model.log_likelihood_counts(x, y)
        assert ll.shape == (10,)
        assert np.all(np.isfinite(ll))

    def test_poisson_ll_high_count_at_high_rate(self, obs_model):
        # High engagement -> high rate -> high count more likely
        x_hi = np.tile(np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9]), (1, 1))
        x_lo = np.tile(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), (1, 1))
        y_hi = np.array([[20, 20, 20, 20]])
        y_lo = np.array([[0, 0, 0, 0]])
        assert obs_model.log_likelihood_counts(x_hi, y_hi)[0] > \
               obs_model.log_likelihood_counts(x_hi, y_lo)[0]
        assert obs_model.log_likelihood_counts(x_lo, y_lo)[0] > \
               obs_model.log_likelihood_counts(x_lo, y_hi)[0]

    def test_poisson_ll_zero_rate_handles_zero_count(self, obs_model):
        # With a very negative log-rate, the expected count is ~0
        x = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (1, 1))
        # Force a very low rate via a custom w_count
        obs_model.params.w_count = -10.0 * np.eye(4, 6)
        y = np.array([[0, 0, 0, 0]])
        ll = obs_model.log_likelihood_counts(x, y)
        assert np.isfinite(ll[0])

    # ---- Bernoulli -------------------------------------------------------
    def test_bernoulli_ll_shape(self, obs_model):
        x = np.tile(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), (10, 1))
        y = np.ones((10, 1), dtype=int)
        ll = obs_model.log_likelihood_binary(x, y)
        assert ll.shape == (10,)
        assert np.all(np.isfinite(ll))

    def test_bernoulli_ll_favors_matching_signal(self, obs_model):
        # w_binary loads onto progress (index 4)
        x_hi = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (1, 1))
        x_lo = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (1, 1))
        y1 = np.array([[1]])
        y0 = np.array([[0]])
        assert obs_model.log_likelihood_binary(x_hi, y1)[0] > \
               obs_model.log_likelihood_binary(x_hi, y0)[0]
        assert obs_model.log_likelihood_binary(x_lo, y0)[0] > \
               obs_model.log_likelihood_binary(x_lo, y1)[0]

    def test_bernoulli_no_nan_on_extremes(self, obs_model):
        # Extreme logits should be clipped, not produce NaN
        x = np.tile(np.array([1e6, 1e6, 1e6, 1e6, 1e6, 1e6]), (1, 1))
        y = np.array([[1]])
        ll = obs_model.log_likelihood_binary(x, y)
        assert np.isfinite(ll[0])

    # ---- Ordered probit --------------------------------------------------
    def test_ordinal_ll_shape(self, obs_model):
        x = np.tile(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), (10, 1))
        y = np.zeros((10, 1), dtype=int)
        ll = obs_model.log_likelihood_ordinal(x, y)
        assert ll.shape == (10,)
        assert np.all(np.isfinite(ll))

    def test_ordinal_ll_higher_eta_shifts_distribution(self, obs_model):
        # The ordinal channel loads onto e^SR (index 3)
        x_lo = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (1, 1))
        x_hi = np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), (1, 1))
        # y=0 (low ordinal) more likely under x_lo; y=3 (high) under x_hi
        y0 = np.array([[0]])
        y3 = np.array([[3]])
        assert obs_model.log_likelihood_ordinal(x_lo, y0)[0] > \
               obs_model.log_likelihood_ordinal(x_lo, y3)[0]
        assert obs_model.log_likelihood_ordinal(x_hi, y3)[0] > \
               obs_model.log_likelihood_ordinal(x_hi, y0)[0]

    def test_ordinal_probabilities_sum_to_one(self, obs_model):
        # Directly check the categorical distribution over y in {0, 1, 2, 3}
        from scipy.special import ndtr
        x = np.tile(np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0]), (1, 1))
        eta = (x @ obs_model.params.w_ordinal.T)[0, 0]
        t = obs_model.params.ordinal_thresholds
        cdf = ndtr(t - eta)
        cum = np.concatenate([[0.0], cdf, [1.0]])
        probs = np.diff(cum)
        assert probs.sum() == pytest.approx(1.0)
        assert np.all(probs >= 0.0)

    def test_ordinal_ll_handles_all_levels(self, obs_model):
        x = np.tile(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), (4, 1))
        y = np.array([[0], [1], [2], [3]])
        ll = obs_model.log_likelihood_ordinal(x, y)
        assert np.all(np.isfinite(ll))

    # ---- Dispatch --------------------------------------------------------
    def test_combined_log_likelihood(self, obs_model):
        x = np.tile(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), (5, 1))
        counts = np.ones((5, 4), dtype=int)
        binaries = np.ones((5, 1), dtype=int)
        ordinals = np.zeros((5, 1), dtype=int)
        ll = obs_model.log_likelihood(x, counts, binaries, ordinals)
        assert ll.shape == (5,)
        # Should equal sum of parts
        parts = (
            obs_model.log_likelihood_counts(x, counts)
            + obs_model.log_likelihood_binary(x, binaries)
            + obs_model.log_likelihood_ordinal(x, ordinals)
        )
        assert np.allclose(ll, parts)

    def test_log_likelihood_missing_channels(self, obs_model):
        x = np.tile(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), (3, 1))
        ll = obs_model.log_likelihood(x)   # all channels None
        assert np.allclose(ll, 0.0)

    def test_log_likelihood_counts_only(self, obs_model):
        x = np.tile(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), (3, 1))
        counts = np.ones((3, 4), dtype=int)
        ll = obs_model.log_likelihood(x, counts=counts)
        assert np.allclose(ll, obs_model.log_likelihood_counts(x, counts))

    # ---- Sampling --------------------------------------------------------
    def test_sample_shapes(self, obs_model, rng):
        x = rng.random((10, N_CONTINUOUS))
        obs = obs_model.sample(x, rng=rng)
        assert obs.counts.shape == (10, 4)
        assert obs.binaries.shape == (10, 1)
        assert obs.ordinals.shape == (10, 1)

    def test_sample_types(self, obs_model, rng):
        x = rng.random((10, N_CONTINUOUS))
        obs = obs_model.sample(x, rng=rng)
        assert obs.counts.dtype.kind in ("i", "u")
        assert obs.binaries.dtype.kind in ("i", "u")
        assert obs.ordinals.dtype.kind in ("i", "u")

    def test_sample_binary_values(self, obs_model, rng):
        x = rng.random((200, N_CONTINUOUS))
        obs = obs_model.sample(x, rng=rng)
        assert set(np.unique(obs.binaries)).issubset({0, 1})

    def test_sample_ordinal_range(self, obs_model, rng):
        x = rng.random((200, N_CONTINUOUS))
        obs = obs_model.sample(x, rng=rng)
        n_levels = len(obs_model.params.ordinal_thresholds) + 1
        assert obs.ordinals.min() >= 0
        assert obs.ordinals.max() < n_levels

    def test_sample_poisson_nonnegative(self, obs_model, rng):
        x = rng.random((200, N_CONTINUOUS))
        obs = obs_model.sample(x, rng=rng)
        assert obs.counts.min() >= 0


# ======================================================================
# Integration
# ======================================================================
class TestIntegration:

    def test_dynamics_and_observation_consistent(self, params, rng):
        """A step through the dynamics, then a draw from the observation
        model, should produce a finite log-likelihood at the next state."""
        d = Dynamics(params)
        om = ObservationModel(params)
        x = rng.random((10, N_CONTINUOUS))
        ev = rng.integers(0, 2, size=(10, 4))
        x_next = d.step(x, ev, rng=rng)
        obs = om.sample(x_next, rng=rng)
        ll = om.log_likelihood(x_next, obs.counts, obs.binaries, obs.ordinals)
        assert np.all(np.isfinite(ll))

    def test_simulate_short_trajectory(self, params, rng):
        """End-to-end sanity: simulate a 5-step trajectory by hand."""
        d = Dynamics(params)
        om = ObservationModel(params)
        x = np.array([[0.1, 0.05, 0.01, 0.2, 0.0, 0.8]])
        s = 0
        for _ in range(5):
            x = d.step(x, rng.integers(0, 2, size=(1, 4)), rng=rng)
            s = int(d.sample_status(x, np.array([s]), rng=rng)[0])
            obs = om.sample(x, rng=rng)
            ll = om.log_likelihood(x, obs.counts, obs.binaries, obs.ordinals)
            assert np.isfinite(ll[0])
        assert 0 <= s < N_STATUS

    def test_deterministic_reproducibility(self, params):
        """Same seed -> identical results across two independent runs."""
        d1 = Dynamics(params)
        d2 = Dynamics(params)
        x = np.tile(np.array([0.3, 0.2, 0.1, 0.4, 0.5, 0.6]), (5, 1))
        ev = np.ones((5, 4))
        a = d1.step(x, ev, rng=np.random.default_rng(7))
        b = d2.step(x, ev, rng=np.random.default_rng(7))
        assert np.allclose(a, b)
import numpy as np
import pytest
import quaternion
from helpers import jk_part, real_bivariate

from bispy import filters

N = 256


@pytest.fixture
def x():
    rng = np.random.default_rng(0)
    return real_bivariate(rng.standard_normal(N), rng.standard_normal(N))


@pytest.fixture
def params():
    rng = np.random.default_rng(1)
    return dict(
        K=1 + rng.random(N),
        eta=0.9 * rng.random(N),
        mu=quaternion.as_quat_array(np.c_[np.zeros(N), rng.standard_normal((N, 3))]),
        alpha=rng.random(N),
        phi=rng.random(N),
    )


def energy(x):
    return np.sum(np.norm(x))


# --- HermitianFilter ------------------------------------------------------------


def test_hermitian_identity(x):
    y = filters.HermitianFilter(N, 1.0, 0.0, quaternion.y).output(x)
    np.testing.assert_allclose(quaternion.as_float_array(y), quaternion.as_float_array(x), atol=1e-12)


def test_hermitian_constant_gain(x):
    y = filters.HermitianFilter(N, 2.0, 0.0, quaternion.y).output(x)
    np.testing.assert_allclose(quaternion.as_float_array(y), 2 * quaternion.as_float_array(x), atol=1e-12)


def test_hermitian_output_is_real_bivariate(x, params):
    f = filters.HermitianFilter(N, params["K"].copy(), params["eta"].copy(), params["mu"])
    assert jk_part(f.output(x)) < 1e-12


def test_hermitian_symmetry_relations(params):
    f = filters.HermitianFilter(N, params["K"].copy(), params["eta"].copy(), params["mu"])
    np.testing.assert_array_equal(f.K[N // 2 + 1 :], f.K[1 : N // 2][::-1])
    np.testing.assert_array_equal(f.eta[N // 2 + 1 :], f.eta[1 : N // 2][::-1])
    inner = np.r_[1 : N // 2, N // 2 + 1 : N]  # DC and Nyquist axes are averages
    np.testing.assert_allclose(np.abs(f.mu[inner]), 1)
    np.testing.assert_array_equal(f.f, np.fft.fftfreq(N))


def test_hermitian_rejects_wrong_sizes(x):
    with pytest.raises(ValueError):
        filters.HermitianFilter(N, np.ones(N - 1), 0.0, quaternion.y)
    f = filters.HermitianFilter(N, 1.0, 0.0, quaternion.y)
    with pytest.raises(ValueError):
        f.output(x[:-1])


def test_hermitian_snapshot(x, params, snapshot):
    f = filters.HermitianFilter(N, params["K"].copy(), params["eta"].copy(), params["mu"], dt=0.5)
    snapshot(y=f.output(x), K=f.K, eta=f.eta, mu=f.mu)


# --- UnitaryFilter --------------------------------------------------------------


def test_unitary_identity(x):
    y = filters.UnitaryFilter(N, quaternion.y, 0.0, 0.0).output(x)
    np.testing.assert_allclose(quaternion.as_float_array(y), quaternion.as_float_array(x), atol=1e-12)


def test_unitary_preserves_energy_and_realness(x, params):
    f = filters.UnitaryFilter(N, params["mu"], params["alpha"].copy(), params["phi"].copy())
    y = f.output(x)
    assert energy(y) == pytest.approx(energy(x), rel=1e-12)
    assert jk_part(y) < 1e-12


def test_unitary_snapshot(x, params, snapshot):
    f = filters.UnitaryFilter(N, params["mu"], params["alpha"].copy(), params["phi"].copy())
    snapshot(y=f.output(x), alpha=f.alpha, phi=f.phi, mu=f.mu)


# --- known bugs (see the audit); remove the marker when fixed --------------------


@pytest.mark.xfail(reason="known bug: odd lengths fail in the symmetry code", raises=ValueError)
@pytest.mark.parametrize("n", [7, 101])
@pytest.mark.parametrize("kind", ["hermitian", "unitary"])
def test_odd_length(kind, n):
    if kind == "hermitian":
        filters.HermitianFilter(n, np.ones(n), 0.5 * np.ones(n), np.ones(n) * quaternion.y)
    else:
        filters.UnitaryFilter(n, np.ones(n) * quaternion.y, np.ones(n), np.ones(n))


@pytest.mark.xfail(reason="known bug: filters modify the caller's parameter arrays in place")
@pytest.mark.parametrize("kind", ["hermitian", "unitary"])
def test_inputs_not_modified(kind, params):
    a, b = params["K"], params["eta"]
    a0, b0 = a.copy(), b.copy()
    if kind == "hermitian":
        filters.HermitianFilter(N, a, b, params["mu"])
    else:
        filters.UnitaryFilter(N, params["mu"], a, b)
    np.testing.assert_array_equal(a, a0)
    np.testing.assert_array_equal(b, b0)

import numpy as np
import pytest
import quaternion
from helpers import as_float

from bispy import utils


@pytest.fixture
def q():
    return quaternion.as_quat_array(np.random.default_rng(0).standard_normal((3, 5, 4)))


# --- symplectic decomposition q = q_1 + i q_2 ---------------------------------


def test_symp_split_synth_round_trip(q):
    q1, q2 = utils.sympSplit(q)
    assert q1.shape == q2.shape == q.shape
    np.testing.assert_array_equal(as_float(utils.sympSynth(q1, q2)), as_float(q))


def test_symp_split_components(q):
    f = as_float(q)
    q1, q2 = utils.sympSplit(q)
    np.testing.assert_array_equal(q1, f[..., 0] + 1j * f[..., 2])
    np.testing.assert_array_equal(q2, f[..., 1] + 1j * f[..., 3])


def test_symp_split_rejects_non_quaternion():
    with pytest.raises(ValueError):
        utils.sympSplit(np.ones(4))


# --- Stokes parameters and geometry ---------------------------------------------


def test_stokes_norm_is_pure_with_squared_modulus(q):
    s = utils.StokesNorm(q)
    np.testing.assert_allclose(as_float(s)[..., 0], 0, atol=1e-12)
    np.testing.assert_allclose(np.abs(s), np.abs(q) ** 2)


def test_single_quaternion_is_fully_polarized(q):
    f = as_float(utils.StokesNorm(q))
    S0 = np.norm(q)
    *_, Phi = utils.Stokes2geo(S0, f[..., 2], f[..., 3], f[..., 1])
    np.testing.assert_allclose(Phi, 1)


@pytest.mark.parametrize("theta", [-1.2, 0.0, 0.4])
@pytest.mark.parametrize("chi", [-0.7, 0.0, 0.3])
def test_geo_stokes_round_trip(theta, chi):
    a, theta_, chi_, Phi = utils.Stokes2geo(*utils.geo2Stokes(1.7, theta, chi))
    np.testing.assert_allclose([a, theta_, chi_, Phi], [1.7, theta, chi, 1.0], atol=1e-12)


def test_stokes2geo_partial_polarization():
    # With Phi < 1, Stokes2geo returns the amplitude of the polarized part, a * sqrt(Phi).
    S = utils.geo2Stokes(2.0, 0.4, -0.3, Phi=0.7)
    a, theta, chi, Phi = utils.Stokes2geo(*S)
    np.testing.assert_allclose([a, theta, chi, Phi], [2.0 * np.sqrt(0.7), 0.4, -0.3, 0.7])


def test_normalize_stokes():
    S0 = np.array([2.0, 4.0]); S1 = np.array([1.0, -2.0]); S2 = np.array([0.0, 2.0]); S3 = np.array([1.0, 0.0])
    S1n, S2n, S3n = utils.normalizeStokes(S0, S1, S2, S3)
    np.testing.assert_allclose([S1n, S2n, S3n], [[0.5, -0.5], [0, 0.5], [0.5, 0]])
    # tol regularizes by tol * max(S0)
    S1n, _, _ = utils.normalizeStokes(S0, S1, S2, S3, tol=0.5)
    np.testing.assert_allclose(S1n, [1 / 4, -2 / 6])


@pytest.mark.filterwarnings("error")
def test_stokes2geo_without_power_or_polarization():
    # no power (S0 = 0): everything undefined; unpolarized (Phi = 0): chi undefined.
    # NaN is returned there, without 0/0 RuntimeWarnings.
    S0, S1, S2, S3 = (np.array(v, dtype=float) for v in ([0, 1, 2], [0, 0, 1], [0, 0, 0], [0, 0, 0]))
    a, theta, chi, Phi = utils.Stokes2geo(S0, S1, S2, S3)
    assert np.isnan([a[0], theta[0], chi[0], Phi[0]]).all()
    assert Phi[1] == 0 and np.isnan(chi[1])
    np.testing.assert_allclose([a[2], theta[2], chi[2], Phi[2]], [1, 0, 0, 0.5])


@pytest.mark.filterwarnings("error")
def test_normalize_stokes_ignores_nan_in_s0():
    # A NaN in S0 (e.g. a masked frequency) only affects its own index.
    S0 = np.array([np.nan, 2.0, 4.0]); S1 = np.array([1.0, 1.0, -2.0]); S2 = S3 = np.zeros(3)
    S1n, _, _ = utils.normalizeStokes(S0, S1, S2, S3)
    assert np.isnan(S1n[0])
    np.testing.assert_allclose(S1n[1:], [0.5, -0.5])
    S1n, _, _ = utils.normalizeStokes(S0, S1, S2, S3, tol=0.5)  # tol * max of the finite S0
    np.testing.assert_allclose(S1n[1:], [1 / 4, -2 / 6])


def test_euler_round_trip():
    rng = np.random.default_rng(3)
    n = 100
    a = rng.uniform(0.1, 3, n)
    theta = rng.uniform(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, n)
    chi = rng.uniform(-np.pi / 4 + 0.01, np.pi / 4 - 0.01, n)
    phi = rng.uniform(-np.pi + 0.01, np.pi - 0.01, n)
    out = utils.quat2euler(utils.euler2quat(a, theta, chi, phi))
    np.testing.assert_allclose(np.array(out), [a, theta, chi, phi], atol=1e-10)


# --- windows --------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rectangle", "hamming", "hanning", "gaussian"])
def test_windows_shape_and_symmetry(name):
    w = getattr(utils.windows, name)(33)
    assert w.shape == (33,)
    np.testing.assert_allclose(w, w[::-1], atol=1e-15)
    assert np.all(w > 0) and np.max(w) <= 1 + 1e-15


def test_window_values():
    np.testing.assert_array_equal(utils.windows.rectangle(4), np.ones(4))
    n = np.arange(1, 9)
    np.testing.assert_allclose(utils.windows.hanning(8), 0.5 - 0.5 * np.cos(2 * np.pi * n / 9))
    np.testing.assert_allclose(utils.windows.hamming(8), 0.54 - 0.46 * np.cos(2 * np.pi * n / 9))
    np.testing.assert_allclose(utils.windows.gaussian(3, sigma=0.1)[[0, 1]], [0.1, 1.0])


def test_gaussian_window_rejects_large_sigma():
    with pytest.raises(ValueError):
        utils.windows.gaussian(8, sigma=0.6)


# --- polarization ellipse -------------------------------------------------------


def test_polarization_axis_is_normalized_stokes_vector():
    mu = utils.polarizationAxis(0.4, -0.3)
    _, S1, S2, S3 = utils.geo2Stokes(1, 0.4, -0.3)
    np.testing.assert_allclose(as_float(mu), [0, S3, S1, S2])
    assert np.isclose(np.abs(mu), 1)


def test_polarization_axis_is_stokes_norm_of_polar_form():
    # X j conj(X) = S0 mu for X = a exp(i theta) exp(-k chi) exp(j phi), whatever phi
    rng = np.random.default_rng(5)
    theta = rng.uniform(-np.pi / 2, np.pi / 2, 20)
    chi = rng.uniform(-np.pi / 4, np.pi / 4, 20)
    q = utils.euler2quat(1.0, theta, chi, rng.uniform(-np.pi, np.pi, 20))
    np.testing.assert_allclose(as_float(utils.polarizationAxis(theta, chi)), as_float(utils.StokesNorm(q)), atol=1e-12)


def test_polarization_axis_broadcasts():
    theta = np.linspace(-1, 1, 7)
    mu = utils.polarizationAxis(theta, 0.2)
    assert mu.shape == (7,) and mu.dtype == np.quaternion
    np.testing.assert_allclose(as_float(mu[3]), as_float(utils.polarizationAxis(theta[3], 0.2)))


def test_polarization_axis_round_trip_through_quaternion_psd():
    from bispy import spectral

    N = 64
    psd = spectral.quaternionPSD(N, np.ones(N), 0.6, utils.polarizationAxis(0.5, 0.25))
    _, theta, chi, Phi = utils.Stokes2geo(psd.S0, psd.S1, psd.S2, psd.S3)
    pos = slice(1, N // 2)
    np.testing.assert_allclose([theta[pos], chi[pos], Phi[pos]], np.broadcast_to([[0.5], [0.25], [0.6]], (3, N // 2 - 1)))


def test_polarization_ellipse_linear_and_circular():
    _, ell = utils.polarizationEllipse(0.3, 0.0, a=2)
    np.testing.assert_allclose(np.imag(ell * np.exp(-0.3j)), 0, atol=1e-12)  # a segment
    _, ell = utils.polarizationEllipse(0.3, np.pi / 4, a=2)
    np.testing.assert_allclose(np.abs(ell), 2 * np.sqrt(0.5))  # a circle

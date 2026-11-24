import numpy as np
import pytest
import quaternion
from helpers import as_float, quiet, real_bivariate

from bispy import spectral

N = 256
DT = 0.01


@pytest.fixture
def t():
    return np.arange(N) * DT


@pytest.fixture
def x():
    rng = np.random.default_rng(0)
    return real_bivariate(rng.standard_normal(N), rng.standard_normal(N))


@pytest.fixture
def x_stable(rs):
    """Input for snapshots: legacy RandomState stream, frozen across numpy versions."""
    return real_bivariate(rs.standard_normal(N), rs.standard_normal(N))


def mean_power(x):
    return np.mean(np.norm(x))


# --- quaternionPSD --------------------------------------------------------------


def make_psd(mu_i=0.0, n=N):
    f = np.fft.fftfreq(n)
    S0 = 1 / (1 + (np.abs(f) / 0.1) ** 2)
    Phi = 0.3 + 0.5 * np.exp(-(((np.abs(f) - 0.2) / 0.05) ** 2))
    ang = 0.5 + 2 * f
    c = np.sqrt(1 - mu_i**2)
    mu = quaternion.as_quat_array(np.c_[np.zeros(n), mu_i * np.ones(n), c * np.cos(ang), c * np.sin(ang)])
    return spectral.quaternionPSD(n, S0, Phi, mu), S0, Phi


@pytest.mark.parametrize("mu_i", [0.0, 0.6])
def test_psd_recovers_parameters(mu_i):
    psd, S0, Phi = make_psd(mu_i)
    half = slice(0, N // 2 + 1)
    np.testing.assert_array_equal(psd.S0[half], S0[half])
    inner = np.r_[1 : N // 2, N // 2 + 1 : N]
    Phi_rec = np.sqrt(psd.S1**2 + psd.S2**2 + psd.S3**2) / psd.S0
    np.testing.assert_allclose(Phi_rec[inner], psd.Phi[inner])
    np.testing.assert_allclose(psd.f, np.fft.fftfreq(N))


def test_psd_no_circular_polarization_at_dc_and_nyquist():
    # A real bivariate signal has S3 = 0 at DC and Nyquist. The constructor enforces
    # this by averaging mu there, which lowers the effective degree of polarization
    # to Phi * |mu| (0.8 instead of 1 here).
    psd, *_ = make_psd(mu_i=0.6)
    assert psd.S3[0] == pytest.approx(0, abs=1e-12)
    assert psd.S3[N // 2] == pytest.approx(0, abs=1e-12)
    assert np.abs(psd.mu[0]) == pytest.approx(0.8)


@pytest.mark.parametrize("n", [N, 255, 7])
def test_psd_hermitian_symmetry(n):
    psd, *_ = make_psd(mu_i=0.6, n=n)
    k = np.arange(1, (n - 1) // 2 + 1)  # all positive frequencies (Nyquist excluded)
    np.testing.assert_allclose(psd.S0[-k], psd.S0[k])
    np.testing.assert_allclose(psd.S1[-k], psd.S1[k])
    np.testing.assert_allclose(psd.S2[-k], psd.S2[k])
    np.testing.assert_allclose(psd.S3[-k], -psd.S3[k])


def test_psd_scalar_parameters():
    psd = spectral.quaternionPSD(N, 2.0, 0.5, quaternion.y)
    np.testing.assert_allclose(psd.S0, 2.0)
    np.testing.assert_allclose(psd.S1, 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [dict(Phix=1.5), dict(Phix=-0.1), dict(mux=np.ones(N)), dict(S0x=np.ones(N - 1))],
)
def test_psd_rejects_invalid_input(kwargs):
    args = dict(N=N, S0x=np.ones(N), Phix=0.5 * np.ones(N), mux=np.ones(N) * quaternion.y)
    args.update(kwargs)
    with pytest.raises(ValueError):
        spectral.quaternionPSD(**args)


def test_psd_snapshot(snapshot):
    psd, *_ = make_psd(mu_i=0.6)
    snapshot(density=psd.density)


@pytest.mark.parametrize("n", [7, 101])
def test_psd_odd_length(n):
    psd, S0, Phi = make_psd(mu_i=0.6, n=n)
    half = slice(0, (n + 1) // 2)  # DC and positive frequencies are kept as given
    np.testing.assert_array_equal(psd.S0[half], S0[half])
    np.testing.assert_array_equal(psd.Phi[half], Phi[half])
    assert psd.S3[0] == pytest.approx(0, abs=1e-12)


# --- Periodogram ----------------------------------------------------------------


def test_periodogram_parseval(t, x):
    with quiet():
        p = spectral.Periodogram(t, x)
    assert np.sum(p.S0) / (N * DT) == pytest.approx(mean_power(x))
    np.testing.assert_allclose(p.f, np.fft.fftfreq(N, d=DT))


def test_periodogram_single_realization_fully_polarized(t, x):
    with quiet():
        p = spectral.Periodogram(t, x)
    p.normalize()
    np.testing.assert_allclose(p.Phi, 1)


def test_periodogram_arithmetic(t, x):
    with quiet():
        p = spectral.Periodogram(t, x)
    np.testing.assert_array_equal(as_float((p + p).density), 2 * as_float(p.density))
    np.testing.assert_array_equal(as_float((3 * p).density), 3 * as_float(p.density))
    np.testing.assert_array_equal(as_float((p * 3).density), 3 * as_float(p.density))
    np.testing.assert_array_equal((p + p).S0, 2 * p.S0)
    with pytest.raises(ValueError):
        p * np.ones(2)


def test_periodogram_rejects_non_quaternion(t):
    with pytest.raises(ValueError):
        spectral.Periodogram(t, np.ones(N))


def test_periodogram_snapshot(t, x_stable, snapshot):
    with quiet():
        p = spectral.Periodogram(t, x_stable)
    snapshot(density=p.density)


# --- Multitaper -----------------------------------------------------------------


@pytest.mark.parametrize("bw", [2.5, 4])
def test_multitaper_properties(t, x, bw):
    with quiet():
        m = spectral.Multitaper(t, x, bw=bw)
    n_tapers = int(np.floor(2 * bw)) - 1
    assert m.densities.shape == (N, n_tapers)
    assert m.dpss.shape == (N, n_tapers)
    np.testing.assert_allclose(np.sum(m.dpss**2, axis=0), 1)  # unit-energy tapers
    m.normalize()
    assert np.all(m.S0 > 0)
    assert np.all(m.Phi <= 1 + 1e-12)
    # averaging over tapers makes the estimate partially polarized
    assert np.max(m.Phi) < 1
    # approximately energy preserving
    assert np.sum(m.S0) / (N * DT) == pytest.approx(mean_power(x), rel=0.05)


def test_multitaper_arithmetic(t, x):
    with quiet():
        m = spectral.Multitaper(t, x)
        np.testing.assert_allclose(as_float((m + m).density), 2 * as_float(m.density))
        np.testing.assert_allclose(as_float((0.5 * m).density), 0.5 * as_float(m.density))


@pytest.mark.parametrize("bw", [2.5, 4])
def test_multitaper_snapshot(t, x_stable, bw, snapshot):
    with quiet():
        m = spectral.Multitaper(t, x_stable, bw=bw)
    snapshot(density=m.density)


# --- known bugs (see the audit); remove the marker when fixed --------------------


@pytest.mark.xfail(reason="known bug: `np.any(...) is True` is always False, so the time check never fires")
@pytest.mark.parametrize("estimator", ["Periodogram", "Multitaper"])
def test_sum_rejects_different_time_arrays(t, x, estimator):
    cls = getattr(spectral, estimator)
    with quiet():
        a, b = cls(t, x), cls(2 * t, x)
    with pytest.raises(ValueError):
        a + b


@pytest.mark.xfail(reason="known bug: Multitaper + and * rebuild the result with the default bw")
def test_multitaper_arithmetic_keeps_bandwidth(t, x):
    with quiet():
        m = spectral.Multitaper(t, x, bw=5)
        s = m + m
    assert s.densities.shape == m.densities.shape

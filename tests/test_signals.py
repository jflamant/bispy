import numpy as np
import pytest
import quaternion
from helpers import as_float, jk_part, quiet

import bispy as bsp
from bispy import signals


@pytest.fixture
def amfm_params():
    t = np.linspace(0, 1, 512, endpoint=False)
    return dict(
        a=1 + 0.2 * np.cos(2 * np.pi * 2 * t),
        theta=0.3 + 0.5 * t,
        chi=0.2 - 0.3 * t,
        phi=2 * np.pi * 40 * t,
    )


# --- bivariateAMFM --------------------------------------------------------------


def test_amfm_closed_form(amfm_params):
    p = amfm_params
    z = signals.bivariateAMFM(**p, Hembedding=False, complexOutput=True)
    expected = p["a"] * np.exp(1j * p["theta"]) * (
        np.cos(p["chi"]) * np.cos(p["phi"]) + 1j * np.sin(p["chi"]) * np.sin(p["phi"])
    )
    np.testing.assert_allclose(z, expected, atol=1e-12)


def test_amfm_output_types_agree(amfm_params):
    z = signals.bivariateAMFM(**amfm_params, Hembedding=False, complexOutput=True)
    x = signals.bivariateAMFM(**amfm_params, Hembedding=False)
    assert x.dtype == "quaternion"
    assert jk_part(x) == 0
    np.testing.assert_array_equal(as_float(x)[:, 0] + 1j * as_float(x)[:, 1], z)


def test_amfm_hembedding_is_euler_form(amfm_params):
    x = signals.bivariateAMFM(**amfm_params)
    np.testing.assert_allclose(as_float(x), as_float(bsp.utils.euler2quat(**amfm_params)))


# --- bivariatewhiteNoise --------------------------------------------------------


def test_white_noise_shapes_and_types(legacy_seed):
    w = signals.bivariatewhiteNoise(100, 1.0)
    assert w.shape == (100,) and w.dtype == "quaternion"
    assert jk_part(w) == 0
    wc = signals.bivariatewhiteNoise(100, 1.0, complexOutput=True)
    assert wc.dtype == complex


def test_white_noise_quaternion_and_complex_agree():
    np.random.seed(5)
    wq = signals.bivariatewhiteNoise(64, 1.0, P=0.3, theta=0.2)
    np.random.seed(5)
    wc = signals.bivariatewhiteNoise(64, 1.0, P=0.3, theta=0.2, complexOutput=True)
    np.testing.assert_array_equal(as_float(wq)[:, 0] + 1j * as_float(wq)[:, 1], wc)


@pytest.mark.parametrize("P", [-0.1, 1.1])
def test_white_noise_rejects_invalid_polarization(P):
    with pytest.raises(ValueError):
        signals.bivariatewhiteNoise(10, 1.0, P=P)


def test_white_noise_statistics(legacy_seed):
    # E|w|^2 = S0 and E[w^2] = S0 P exp(2 i theta) (complex representation w = u + i v)
    w = signals.bivariatewhiteNoise(200_000, 2.0, P=0.6, theta=0.4, complexOutput=True)
    power = np.mean(np.abs(w) ** 2)
    m2 = np.mean(w**2)
    assert power == pytest.approx(2.0, rel=0.01)
    assert np.abs(m2) / power == pytest.approx(0.6, abs=0.01)
    assert np.angle(m2) / 2 == pytest.approx(0.4, abs=0.01)


def test_white_noise_snapshot(legacy_seed, snapshot):
    snapshot(w=signals.bivariatewhiteNoise(64, 1.5, P=0.4, theta=0.3))


# --- stationaryBivariate --------------------------------------------------------


@pytest.fixture
def target_psd():
    N = 256
    f = np.fft.fftfreq(N)
    S0 = 1 / (1 + (np.abs(f) / 0.1) ** 2)
    Phi = 0.3 + 0.5 * np.exp(-(((np.abs(f) - 0.2) / 0.05) ** 2))
    mu = quaternion.as_quat_array(np.c_[np.zeros(N), np.zeros(N), np.cos(0.5) * np.ones(N), np.sin(0.5) * np.ones(N)])
    return bsp.spectral.quaternionPSD(N, S0, Phi, mu)


def test_stationary_bivariate_rejects_non_psd():
    with pytest.raises(ValueError):
        signals.stationaryBivariate(np.ones(8))


def test_stationary_bivariate_simulate_shape(target_psd, legacy_seed):
    sim = signals.stationaryBivariate(target_psd)
    assert sim.simulation is None
    sim.simulate(3)
    assert sim.simulation.shape == (3, target_psd.N)
    assert jk_part(sim.simulation) < 1e-12  # real bivariate realizations


def test_stationary_bivariate_odd_length(legacy_seed):
    n = 101
    mu = np.ones(n) * quaternion.quaternion(0, 0.5, 1, 0)
    sim = signals.stationaryBivariate(bsp.spectral.quaternionPSD(n, np.ones(n), 0.5 * np.ones(n), mu))
    sim.simulate(2)
    assert sim.simulation.shape == (2, n)
    assert jk_part(sim.simulation) < 1e-12


def test_stationary_bivariate_matches_target_psd(target_psd, legacy_seed):
    # The average periodogram of many realizations approaches the target PSD.
    M, N = 300, target_psd.N
    sim = signals.stationaryBivariate(target_psd)
    sim.simulate(M)
    t = np.arange(N, dtype=float)
    dens = np.zeros((N, 4))
    for m in range(M):
        with quiet():
            dens += as_float(bsp.spectral.Periodogram(t, sim.simulation[m]).density)
    dens /= M
    target = as_float(target_psd.density)
    ratio = dens[:, 0] / target[:, 0]
    assert np.median(ratio) == pytest.approx(1, abs=0.03)
    rel_err = np.median(np.abs(dens - target)) / np.median(target[:, 0])
    assert rel_err < 0.1


def test_stationary_bivariate_snapshot(target_psd, legacy_seed, snapshot):
    sim = signals.stationaryBivariate(target_psd)
    sim.simulate(2)
    snapshot(simulation=sim.simulation, K=sim.K, eta=sim.eta, mu=sim.mu)

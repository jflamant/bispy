"""Smoke tests: every plotting entry point runs and returns figure/axes handles."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import quaternion
from helpers import two_chirps

import bispy as bsp
from bispy import timefrequency as tf


@pytest.fixture(scope="module")
def chirps():
    return two_chirps(256)


@pytest.fixture(scope="module")
def qstft(chirps):
    t, x = chirps
    S = tf.QSTFT(x, t)
    S.compute(nperseg=51, noverlap=48, nfft=256, ridges=True)
    return S


@pytest.fixture(scope="module")
def qcwt(chirps):
    t, x = chirps
    C = tf.QCWT(x, t)
    C.compute(5, 120, dict(type="Morse", beta=12, gamma=3), Nscales=32, ridges=True)
    return C


@pytest.fixture(scope="module")
def psd():
    N = 128
    mu = np.ones(N) * quaternion.quaternion(0, 0, 1, 1)
    return bsp.spectral.quaternionPSD(N, np.ones(N), 0.5 * np.ones(N), mu)


def is_figure(fig):
    return isinstance(fig, plt.Figure)


@pytest.mark.parametrize("kind", ["plot2D", "plot3D"])
def test_visual(chirps, kind):
    t, x = chirps
    fig, ax = getattr(bsp.utils.visual, kind)(t, x)
    assert is_figure(fig)


@pytest.mark.parametrize("kind", ["2D", "3D"])
def test_plot_signal(qstft, kind):
    fig, ax = qstft.plotSignal(kind=kind)
    assert is_figure(fig)


@pytest.mark.parametrize("single_sided", [True, False])
def test_qstft_plot_stokes(qstft, single_sided):
    fig, ax = qstft.plotStokes(single_sided=single_sided)
    assert is_figure(fig) and len(ax) == 4


def test_qstft_plot_ridges(qstft):
    fig, ax = qstft.plotRidges(quivertdecim=10)
    assert is_figure(fig) and len(ax) == 3


def test_qcwt_plot_stokes(qcwt):
    fig, ax = qcwt.plotStokes()
    assert is_figure(fig) and len(ax) == 4


def test_qcwt_plot_ridges(qcwt):
    fig, ax = qcwt.plotRidges(quivertdecim=10)
    assert is_figure(fig) and len(ax) == 3


@pytest.mark.parametrize("which", ["qstft", "qcwt"])
def test_plot_ridges_when_no_ridge_is_found(qstft, qcwt, which, monkeypatch):
    obj = {"qstft": qstft, "qcwt": qcwt}[which]
    monkeypatch.setattr(obj, "ridges", [])
    monkeypatch.setattr(obj, "extractRidges", lambda *args, **kwargs: None)  # finds nothing
    fig, ax = obj.plotRidges()
    assert is_figure(fig)
    assert len(ax[2].collections) == 0  # no ridge drawn, but the figure is complete


@pytest.mark.parametrize("estimator", ["Periodogram", "Multitaper"])
def test_spectral_estimate_plot(chirps, estimator):
    t, x = chirps
    est = getattr(bsp.spectral, estimator)(t, x)
    est.normalize(tol=0.01)
    fig, axes = est.plot()
    assert is_figure(fig) and len(axes) == 6


@pytest.mark.parametrize("method", ["plot", "plotStokes"])
@pytest.mark.parametrize("single_sided", [True, False])
def test_psd_plots(psd, method, single_sided):
    fig, ax = getattr(psd, method)(single_sided=single_sided)
    assert is_figure(fig)


def test_psd_plot_two_sided_frequency_axis(psd):
    fig, ax = psd.plot(single_sided=False)
    lo, hi = ax[2].get_xlim()
    assert lo <= psd.f.min() and hi >= psd.f.max() - 1e-12


def test_plot2d_complex_input(chirps):
    t, _ = chirps
    z = np.exp(2j * np.pi * 5 * t)
    fig, ax = bsp.utils.visual.plot2D(t, z)
    assert is_figure(fig)
    np.testing.assert_array_equal(ax[0].lines[0].get_ydata(), z.real)
    np.testing.assert_array_equal(ax[1].lines[0].get_ydata(), z.imag)


@pytest.mark.parametrize(
    "which", ["qstft.plotStokes", "qcwt.plotStokes", "qstft.plotRidges", "psd.plot", "psd.plotStokes"]
)
def test_plots_leave_global_rng_untouched(qstft, qcwt, psd, which):
    obj, method = which.split(".")
    plot = getattr({"qstft": qstft, "qcwt": qcwt, "psd": psd}[obj], method)
    np.random.seed(0)
    expected = np.random.random()
    np.random.seed(0)
    plot()
    assert np.random.random() == expected

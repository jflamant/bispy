import numpy as np
import pytest
from helpers import as_float, jk_part, real_bivariate, two_chirps

import bispy as bsp
from bispy import timefrequency as tf


@pytest.fixture
def chirp():
    """Single polarized linear chirp, f_inst(t) = 50 + 100 t, theta = 0.4, chi = 0.2."""
    N = 512
    t = np.linspace(0, 1, N, endpoint=False)
    env = bsp.utils.windows.hanning(N)
    x = bsp.signals.bivariateAMFM(env, 0.4, 0.2, 2 * np.pi * (50 * t + 50 * t**2), Hembedding=False)
    return t, x


@pytest.fixture
def tone():
    """Polarized tone at 60 Hz, theta = 0.2, chi = 0.1."""
    N = 512
    t = np.arange(N) / N
    return t, bsp.signals.bivariateAMFM(np.ones(N), 0.2, 0.1, 2 * np.pi * 60 * t, Hembedding=False)


def ellipse_params(S1n, S2n, S3n):
    theta = 0.5 * np.arctan2(S2n, S1n)
    chi = 0.5 * np.arcsin(np.clip(S3n / np.sqrt(S1n**2 + S2n**2 + S3n**2), -1, 1))
    return theta, chi


# --- Hembedding -----------------------------------------------------------------


@pytest.fixture
def amfm():
    t = np.linspace(0, 1, 512, endpoint=False)
    p = dict(a=1 + 0.2 * np.cos(2 * np.pi * 2 * t), theta=0.3 + 0.5 * t, chi=0.2 - 0.3 * t, phi=2 * np.pi * 40 * t)
    return p, bsp.signals.bivariateAMFM(**p, Hembedding=False)


def test_hembedding_keeps_original_signal(amfm):
    _, x = amfm
    H = tf.Hembedding(x)
    np.testing.assert_allclose(as_float(H.Hembedding)[:, :2], as_float(x)[:, :2], atol=1e-12)


def test_hembedding_recovers_instantaneous_parameters(amfm):
    p, x = amfm
    H = tf.Hembedding(x)
    mid = slice(64, -64)  # away from the edges
    np.testing.assert_allclose(H.a[mid], p["a"][mid], atol=5e-3)
    np.testing.assert_allclose(H.theta[mid], p["theta"][mid], atol=1e-2)
    np.testing.assert_allclose(H.chi[mid], p["chi"][mid], atol=1e-2)


@pytest.mark.parametrize("n", [128, 127, 9])
def test_hembedding_any_length(n):
    rng = np.random.default_rng(n)
    x = real_bivariate(rng.standard_normal(n), rng.standard_normal(n))
    H = tf.Hembedding(x).Hembedding
    np.testing.assert_allclose(as_float(H)[:, :2], as_float(x)[:, :2], atol=1e-12)
    # no negative frequencies (bins n//2+1..n-1; Nyquist is kept)
    np.testing.assert_allclose(as_float(bsp.qfft.Qfft(H))[n // 2 + 1 :], 0, atol=1e-12)


def test_hembedding_rejects_non_quaternion():
    with pytest.raises(ValueError):
        tf.Hembedding(np.ones(8))


def test_hembedding_snapshot(rs, snapshot):
    x = real_bivariate(rs.standard_normal(128), rs.standard_normal(128))
    snapshot(hembedding=tf.Hembedding(x).Hembedding)


# --- QSTFT ----------------------------------------------------------------------


def test_qstft_accepts_complex_input(chirp):
    t, x = chirp
    z = as_float(x)[:, 0] + 1j * as_float(x)[:, 1]
    S = tf.QSTFT(z, t)
    np.testing.assert_array_equal(as_float(S.x), as_float(x))


def test_qstft_default_time_axis(chirp):
    _, x = chirp
    np.testing.assert_array_equal(tf.QSTFT(x).t, np.arange(len(x)))


def test_qstft_perfect_reconstruction(chirp):
    t, x = chirp
    S = tf.QSTFT(x, t)
    S.compute(nperseg=101, noverlap=96, nfft=len(x))
    _, xr = S.inverse()
    assert len(xr) >= len(x)  # scipy's istft returns the padded length
    np.testing.assert_allclose(as_float(xr[: len(x)]), as_float(x), atol=1e-12)
    assert jk_part(xr) < 1e-12


def test_qstft_masked_inverse_removes_energy(chirp):
    t, x = chirp
    S = tf.QSTFT(x, t)
    S.compute(nperseg=101, noverlap=96, nfft=len(x))
    _, xr = S.inverse(mask=np.zeros(S.S0.shape, dtype=bool))
    np.testing.assert_allclose(as_float(xr), 0, atol=1e-12)


def test_qstft_fully_polarized_pointwise(chirp):
    t, x = chirp
    S = tf.QSTFT(x, t)
    S.compute(nperseg=101, noverlap=96, nfft=len(x))
    np.testing.assert_allclose(np.sqrt(S.S1**2 + S.S2**2 + S.S3**2), S.S0, atol=1e-12 * S.S0.max())
    assert np.max(np.sqrt(S.S1n**2 + S.S2n**2 + S.S3n**2)) <= 1


def test_qstft_ridge_follows_chirp(chirp):
    t, x = chirp
    S = tf.QSTFT(x, t)
    S.compute(nperseg=101, noverlap=96, nfft=len(x), ridges=True)
    assert len(S.ridges) == 1
    fr, tr = np.array(S.ridges[0][0]), np.array(S.ridges[0][1])
    ft, tt = S.f[fr], S.sampled_times[tr]
    inside = (tt > 0.2) & (tt < 0.8)
    assert np.max(np.abs(ft - (50 + 100 * tt))[inside]) < 2  # within 2 frequency bins
    theta, chi = ellipse_params(S.S1n[fr, tr], S.S2n[fr, tr], S.S3n[fr, tr])
    np.testing.assert_allclose(theta[inside], 0.4, atol=5e-3)
    np.testing.assert_allclose(chi[inside], 0.2, atol=5e-3)


def test_qstft_two_chirps_two_ridges():
    t, x = two_chirps(1024)
    S = tf.QSTFT(x, t)
    S.compute(nperseg=101, noverlap=100, nfft=1024)
    S.extractRidges()
    assert len(S.ridges) == 2


def test_qstft_snapshot(rs, snapshot):
    N = 128
    t = np.arange(N) * 0.01
    x = real_bivariate(rs.standard_normal(N), rs.standard_normal(N))
    S = tf.QSTFT(x, t)
    S.compute(nperseg=31, noverlap=24, nfft=N)
    snapshot(f=S.f, times=S.sampled_times, S0=S.S0, S1=S.S1, S2=S.S2, S3=S.S3)


# --- QCWT -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "wavelet",
    [dict(type="Morse", beta=12, gamma=3), dict(type="Morse"), dict(type="Morlet")],
    ids=["morse-12-3", "morse-default", "morlet"],
)
def test_qcwt_locates_tone_and_polarization(tone, wavelet):
    t, x = tone
    C = tf.QCWT(x, t)
    C.compute(10, 200, wavelet, Nscales=40)
    assert C.tfpr.shape == (40, len(t))
    k = np.argmax(C.S0[:, len(t) // 2])
    step = np.log(C.sampled_frequencies[1] / C.sampled_frequencies[0])
    assert abs(np.log(C.sampled_frequencies[k] / 60)) <= step
    mid = slice(100, -100)
    theta, chi = ellipse_params(C.S1n[k, mid], C.S2n[k, mid], C.S3n[k, mid])
    np.testing.assert_allclose(theta, 0.2, atol=1e-10)
    np.testing.assert_allclose(chi, 0.1, atol=1e-10)


def test_qcwt_morse_bandpass_normalization(tone):
    t, x = tone
    C = tf.QCWT(x, t)
    C.compute(10, 200, dict(type="Morse", beta=12, gamma=3), Nscales=40)
    peaks = np.max(as_float(C.params["wavelets"])[..., 0], axis=1)
    assert np.all(peaks <= 2 + 1e-12) and np.all(peaks > 1.9)  # peak value 2 at the center frequency


def test_qcwt_rejects_unknown_wavelet(tone):
    t, x = tone
    with pytest.raises(ValueError):
        tf.QCWT(x, t).compute(10, 200, dict(type="Haar"))


def test_qcwt_ridges():
    t, x = two_chirps(512)
    C = tf.QCWT(x, t)
    C.compute(5, 250, dict(type="Morse", beta=12, gamma=3), Nscales=64, ridges=True)
    assert len(C.ridges) >= 1


@pytest.mark.parametrize(
    "wavelet", [dict(type="Morse", beta=12, gamma=3), dict(type="Morlet")], ids=["morse", "morlet"]
)
def test_qcwt_snapshot(rs, wavelet, snapshot):
    # The Morlet snapshot changes when the normalization bug is fixed (S0 x 4).
    N = 128
    t = np.arange(N) / N
    x = real_bivariate(rs.standard_normal(N), rs.standard_normal(N))
    C = tf.QCWT(x, t)
    C.compute(4, 50, wavelet, Nscales=16)
    snapshot(freqs=C.sampled_frequencies, S0=C.S0, S1=C.S1, S2=C.S2, S3=C.S3)


# --- known bugs (see the audit); remove the marker when fixed --------------------


def test_qstft_ridges_with_default_nfft(chirp):
    # nfft=None means nfft=nperseg (scipy): both must give the same ridges.
    t, x = chirp
    default, explicit = tf.QSTFT(x, t), tf.QSTFT(x, t)
    default.compute(nperseg=101, noverlap=96)
    default.extractRidges()
    explicit.compute(nperseg=101, noverlap=96, nfft=101)
    explicit.extractRidges()
    assert len(default.ridges) == 1
    assert default.ridges == explicit.ridges


@pytest.mark.parametrize("eta", [None, 1.0, 4.0])
def test_qcwt_morlet_normalization(tone, eta):
    t, x = tone
    params = dict(type="Morlet") if eta is None else dict(type="Morlet", eta=eta)
    eta = 2 * np.pi if eta is None else eta
    C = tf.QCWT(x, t)
    C.compute(10, 200, params, Nscales=8)
    f = np.fft.fftfreq(len(t), d=t[1] - t[0])
    for k, fs in enumerate(C.sampled_frequencies):
        s = eta / (2 * np.pi * fs)
        c = np.pi ** (-0.25) * (1 + np.exp(-(eta**2)) - 2 * np.exp(-0.75 * eta**2)) ** (-0.5) * np.sqrt(s)
        expected = c * (np.exp(-0.5 * (s * 2 * np.pi * f - eta) ** 2) - np.exp(-0.5 * ((s * 2 * np.pi * f) ** 2 + eta**2)))
        np.testing.assert_allclose(as_float(C.params["wavelets"][k])[:, 0], expected, rtol=1e-10)

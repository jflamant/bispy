import numpy as np
import pytest
import quaternion
from helpers import as_float

from bispy import qfft


@pytest.fixture
def q():
    return quaternion.as_quat_array(np.random.default_rng(0).standard_normal((64, 4)))


def test_inverse_round_trip(q):
    np.testing.assert_allclose(as_float(qfft.iQfft(qfft.Qfft(q))), as_float(q), atol=1e-12)


def test_forward_round_trip(q):
    np.testing.assert_allclose(as_float(qfft.Qfft(qfft.iQfft(q))), as_float(q), atol=1e-12)


def test_linearity(q):
    p = quaternion.as_quat_array(np.random.default_rng(1).standard_normal((64, 4)))
    lhs = qfft.Qfft(2.5 * q + p)
    rhs = 2.5 * qfft.Qfft(q) + qfft.Qfft(p)
    np.testing.assert_allclose(as_float(lhs), as_float(rhs), atol=1e-12)


def test_parseval(q):
    # sum |x|^2 = sum |X|^2 / N (np.norm is the squared modulus)
    X = qfft.Qfft(q)
    np.testing.assert_allclose(np.sum(np.norm(q)), np.sum(np.norm(X)) / q.size)


def test_matches_complex_fft_for_complex_signals():
    # For a (1, j)-complex signal x = a + j b, the QFT is the ordinary FFT.
    rng = np.random.default_rng(2)
    z = rng.standard_normal(32) + 1j * rng.standard_normal(32)
    xq = quaternion.as_quat_array(np.c_[z.real, np.zeros(32), z.imag, np.zeros(32)])
    Z = np.fft.fft(z)
    X = as_float(qfft.Qfft(xq))
    np.testing.assert_allclose(X[:, 0] + 1j * X[:, 2], Z, atol=1e-12)
    np.testing.assert_allclose(X[:, [1, 3]], 0, atol=1e-12)


def test_shift_unshift(q):
    np.testing.assert_array_equal(as_float(qfft.iQfftshift(qfft.Qfftshift(q))), as_float(q))


def test_fftfreq():
    np.testing.assert_array_equal(qfft.Qfftfreq(10, dt=0.1), np.fft.fftfreq(10, d=0.1))

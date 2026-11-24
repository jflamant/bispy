"""bispy reports progress through `logging` (silent by default) and never prints or shows figures."""

import logging

from helpers import two_chirps

import bispy as bsp


def run_everything():
    """Every code path that used to print or call fig.show()."""
    t, x = two_chirps(256)
    S = bsp.timefrequency.QSTFT(x, t)
    S.compute(nperseg=51, noverlap=48, nfft=256)
    S.plotRidges()  # extracts the ridges first
    C = bsp.timefrequency.QCWT(x, t)
    C.compute(5, 120, dict(type="Morse"), Nscales=16, ridges=True)
    for estimator in (bsp.spectral.Periodogram(t, x), bsp.spectral.Multitaper(t, x)):
        estimator.normalize(tol=0.01)
        estimator.plot()
    bsp.utils.visual.plot3D(t, x)


def test_nothing_printed(capsys):
    run_everything()
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""


def test_progress_is_logged(caplog):
    with caplog.at_level(logging.INFO, logger="bispy"):
        run_everything()
    assert "Number of data tapers: 4" in caplog.messages
    assert "No ridges extracted yet, extracting them first" in caplog.messages
    assert any(m.endswith("ridges extracted") for m in caplog.messages)
    assert all(r.name.startswith("bispy.") for r in caplog.records)


def test_figures_are_not_shown(recwarn):
    # fig.show() under a non-interactive backend emits a UserWarning
    run_everything()
    assert not [w for w in recwarn if "show" in str(w.message)]

import matplotlib

# Headless backend: tests must never open windows (CI has no display).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from helpers import as_float  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def legacy_seed():
    """Seed numpy's global RNG, which bispy.signals uses, and restore it afterwards.

    The legacy ``np.random`` stream is frozen by NumPy's compatibility policy,
    so seeded outputs are identical across numpy versions and platforms.
    """
    state = np.random.get_state()
    np.random.seed(20261124)
    yield
    np.random.set_state(state)


@pytest.fixture
def rs():
    """Stable random source for snapshot inputs (legacy RandomState stream)."""
    return np.random.RandomState(1234)


@pytest.fixture
def snapshot(ndarrays_regression):
    """Compare arrays with reference values stored in tests/<module>/<test>.npz.

    Quaternion arrays are stored as float arrays. The tolerance is relative to each
    array's largest value, so snapshots survive FFT/BLAS round-off across platforms
    and dependency versions. After an intentional numerical change, regenerate with
    ``uv run pytest --force-regen`` and commit the updated .npz files.
    """

    def check(**arrays):
        data, tolerances = {}, {}
        for key, value in arrays.items():
            value = as_float(value)
            data[key] = value
            scale = float(np.max(np.abs(value))) if value.size else 0.0
            tolerances[key] = dict(atol=1e-9 * scale, rtol=1e-7)
        ndarrays_regression.check(data, tolerances=tolerances)

    return check

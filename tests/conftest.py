import matplotlib

# Headless backend: tests must never open windows (CI has no display).
matplotlib.use("Agg")

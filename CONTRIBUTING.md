# Contributing to REPEAT

Thanks for considering a contribution!

## Development setup

```bash
git clone https://github.com/Wickstrom/REPEAT.git
cd REPEAT
python -m venv .venv
source .venv/bin/activate
pip install --editable .[dev]
```

REPEAT depends on `relax-xai>=0.2.0`. Until that version is available on
PyPI, install it from the RELAX repository:

```bash
pip install "relax-xai @ git+https://github.com/Wickstrom/RELAX.git"
```

## Running tests and linting

```bash
pytest
ruff check .
ruff format --check .
```

Please make sure both pass before opening a pull request. New features should
come with tests (CPU-friendly, small tensors). The tests in
`tests/test_thresholds.py` check the torch thresholding implementations
against their scikit-image counterparts, and are skipped automatically if
scikit-image is not installed.

## Release process

1. Bump `__version__` in `src/repeat_xai/__init__.py`.
2. Commit, then tag the release and push the tag:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

3. The `publish.yml` workflow builds the package and publishes it to
   [PyPI](https://pypi.org/project/repeat-xai/) using
   [trusted publishing](https://docs.pypi.org/trusted-publishers/). No API
   tokens are needed, but the publisher must be configured on PyPI once:
   add a *pending* publisher for `Wickstrom/REPEAT` with workflow
   `publish.yml` and environment `pypi` (or configure it after the first
   release attempt). Alternatively, create a GitHub environment named `pypi`
   with a `PYPI_API_TOKEN` secret and add `password: ${{ secrets.PYPI_API_TOKEN }}`
   to the `pypa/gh-action-pypi-publish` step.
4. Write release notes on the GitHub release for the tag.

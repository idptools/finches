# Development, testing, and deployment tools

This directory holds the tools used for continuous integration (CI) and for setting up development environments. Nothing here is needed to use FINCHES.

## Manifest

### Conda environment

* `conda-envs/test_env.yaml`: the environment the CI workflow (`.github/workflows/CI.yaml`) builds on every platform and Python version it tests. It provides Python, the build and runtime dependencies that are available from conda-forge, and pytest; the CI then installs the remaining dependencies (metapredict and its PyTorch stack) with `pip install .` and builds the Cython extension in place.

### Scripts

* `scripts/create_conda_env.py`: helper that creates a conda environment from an environment file for a chosen Python version, for example `python devtools/scripts/create_conda_env.py -n finches-test -p 3.11 devtools/conda-envs/test_env.yaml`.

## Running the tests locally

The tests run against the source tree, because their reference data (`finches/tests/test_data/*.npz`) and example structures (`examples/`) are not part of the installed package. From the repository root:

```bash
pip install -e .
python setup.py build_ext --inplace   # compile finches/utils/matrix_manipulation.pyx
pytest finches/tests
```

## How to contribute changes

- Clone the repository if you have write access to the main repo, or fork it if you are a collaborator.
- Make a new branch with `git checkout -b {your branch name}`.
- Make your changes and run the tests.
- Keep the test environment (`conda-envs/test_env.yaml`) in line with the runtime dependencies declared in `pyproject.toml`.
- Push the branch with `git push -u origin {your branch name}` (`origin` is the default name for the remote; yours may differ).
- Open a pull request on GitHub against `develop`.

## Checklist for updates

- [ ] Make sure there is an issue open for your update
- [ ] Create the PR, referencing the issue
- [ ] Debug the PR until the tests pass
- [ ] Tag the final, debugged version with `git tag -a X.Y.Z [latest pushed commit] && git push --follow-tags`
- [ ] Get the PR merged

## Versioning

The version is set from git by [versioningit](https://github.com/jwodder/versioningit), configured under `[tool.versioningit]` in `pyproject.toml`, and written to `finches/_version.py` at build time. A commit that carries a tag `X.Y.Z` builds as version `X.Y.Z`; a later commit builds as `X.Y.Z+N.gHASH`, where `N` is the number of commits since the tag and `HASH` is the short commit hash, with `.dirty` appended if the working tree has uncommitted changes.

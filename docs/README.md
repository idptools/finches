# Building the finches documentation

The docs are built with [Sphinx](http://www.sphinx-doc.org/en/master/) using the Read the Docs theme. To build them locally, make sure Sphinx and the theme are installed in the same environment as finches (the API page uses autodoc, so finches itself has to be importable):

```bash
pip install sphinx sphinx_rtd_theme
```

or create the conda environment described in `docs/requirements.yaml`.

Then, from this directory, build the static HTML with

```bash
make html
```

The compiled pages land in `_build/html/`; open `_build/html/index.html` to view them. `_build/` is ignored by git and should not be committed.

The hosted documentation at [https://finches.readthedocs.io](https://finches.readthedocs.io) is built by Read the Docs from the `.readthedocs.yml` file in the top level of the repository, which points at `docs/conf.py` and installs the packages listed in `docs/requirements.txt`. Heavy dependencies that are not needed to render the docs (metapredict, matplotlib, scipy, the Cython extension, and so on) are mocked out via `autodoc_mock_imports` in `conf.py`, so if you add a new third-party import to a module that autodoc touches, add it to that list as well or the Read the Docs build will fail to import the module.

# Makefile
.PHONY: version build

version:
    versioningit write-version-file

build: version
    python -m build
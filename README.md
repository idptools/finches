finches
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/finches/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/finches/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/finches/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/finches/branch/main)

Epsilon implementation

### Installation


The installation below has been tested in a clean conda environment using Python 3.9; YMMV in other systems. This does not use anything different from our usual stack (`soursop`, `mdtraj`, `numpy`, `scipy` `cython` etc.) so *should* probably install easily into your "default" environment, although I did test ona  totally clean conda environment to be safe!

First install `cython`, `numpy` and `pytorch` using CONDA. 

> NB: This is currently very important because on macOS we need to ensure we're using consistent numpy/pytorch versions from conda OR PyPI, but cannot mix and match.

Specificaly, we recommend the following install instructions:

	# ensure dependencies are from the same ecosystem (conda)
	conda install numpy pytorch scipy cython matplotlib  -c pytorch
	
	# required for integrated disorder prediction
	pip instal metapredict 
	

Then clone finches

	git clone git@github.com:idptools/finches.git
		
I recommend this so you can pull and re-install as I push updates! Finally, install by running the following comamnd instead the main `finches` directory (i.e. where `pyproject.toml` is):

	pip install -e .
	
NB: use `-e` flag because there's something off with the packaging right now and the cython code is not correctly being distributed on the non `-e` install (Alex to fix!).

To check this has worked, run the following command from the terminal 

	python -c  "from finches.frontend import mpipi_frontend"	
### Copyright

Copyright (c) 2023, Garrett M. Ginell & Alex S. Holehouse


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.

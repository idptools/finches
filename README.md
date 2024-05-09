finches
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/finches/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/finches/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/finches/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/finches/branch/main)

![Finches Logo](finches_logo_v1.png)


Epsilon implementation for intermolecular interactions!

### Installation

The installation below has been tested in a clean conda environment using Python 3.9; YMMV in other systems. This does not use anything different from our usual stack (`soursop`, `mdtraj`, `numpy`, `scipy` `cython` etc.) so *should* probably install easily into your "default" environment, although I did test on a totally clean conda environment to be safe!

#### Create new conda environment (optional)
If creating a new conda environment the following command works well (note you can run this anywhere as all the files are created in the conda envs directory):

	conda create -n finches  python=3.9 -y
	
Then activate this environment

	conda activate finches
	
From here we can then install the various dependencies 	
	
#### Install dependencies with conda	

First, install `cython`, `numpy`, and `pytorch` using CONDA. 

> NB: This is **very important** because on macOS, we need to ensure we're using consistent numpy/pytorch versions from conda OR PyPI, but we cannot mix and match.

Specifically, we recommend the following install instructions (this again can be run anywhere):

	# ensure dependencies are from the same ecosystem (conda)
	conda install numpy pytorch scipy cython matplotlib jupyter  -c pytorch
	
	# we do this separately and last
	conda install mdtraj
	
	# required for integrated disorder prediction; note metapredict is
	# only available via pip
	pip install metapredict 

#### Clone finches
Then clone finches; note this SHOULD be executed in a sensible location because this command will clone (copy) the finches Git repo to a local location. Note you will need access to https://github.com/idptools/finches for this, which means you need to be added as a member to idptools. If you've not yet been added LMK!

	git clone git@github.com:idptools/finches.git
		
I recommend this so you can pull and re-install as I push updates! 


#### Install finches
Finally, move into the main finches directory after cloning

	cd finches

And then install by running the following command inside that directory (i.e. where `pyproject.toml` is):

	pip install -e .
	
Once installed, finches will be available in the environment with which you installed it

**NB**: We use the `-e` flag because it locally links to the code in this directory. This is useful for two reasons:

1. If updates are pushed, you just need to run

	 	git pull
	 	
	 in this directory and your live version of the code will be updated the next time you use finches. Given finches is in active development this is especially useful
	 
2. It correctly compiles the associated cython code, and right now there's something off with the packaging right now, and the Cython code is not correctly distributed on a non `-e` install (Alex to fix!).

#### Check it worked
To check this has worked, run the following command in the terminal:

	python -c  "from finches.frontend.mpipi_frontend import Mpipi_frontend; print('Success')"
	
Note this may take a second to run the first time you launch it... If this print's "success" then you're good to go! If not something is up...


### Demo
Head on over to `demo/` directory for some jupyter notebooks showing the types of things you can do with Epsilon/finches.

### Changelog

* **May 9th 2024** - fixed a stupid off-by-one bug in the indexing from the sliding window epsilon functions - basically invisible unless you're looking at matrices of ~10 residues or smaller, but even so...
* **May 5th 2024** - major update and breaking changes; **PLEASE READ**. We realized that the CALVADOS integral for calculating interaction parameters was integrating over nanometers instead of over angstroms, meaning all the CALAVDOS residue-specific interaction parameters were (consistently) smaller than they should be by a d(distance)-factor of 10. This changes nothing about any scientific insights that have been gleaned but does change the absolute numerical values of CALVADOS-derived epsilon analysis. This has now been fixed, but in the process, we used this as an opportunity to update a range of additional things, listed below:
	* As mentioned - CALVADOS epsilon parameters are now ~10x what they were before, which is more in-keeping with epsilon values. This will necessitate to any code that analyses or visualizes CALVADOS-associated epsilon values to update thresholds etc, but the actual rank order and relative epsilon values should all remain approximately the same (we had to make some very small changes to the baseline correction).
	* We shifted to ensure module names are all lowercase (e.g., `finches.forcefields.mPiPi` -> `finches.forcefields.mpipi` to be consistent with Python best practices
	* We shifted `Interaction_Matrix_Constructor` to `InteractionMatrixConstructor` to again be consistent with Python best practices (camel-case for Class names).
	* We unified Mpipi type-setting as "Mpipi", instead of "mPiPi", which had been used previously. This has changed the names of classes, parameter files, etc etc, but everything within finches *should* now be internally consistent. However, existing code may be importing now deprecated parameter names or module names. This should be an easy fix in your local code.
	* We removed the `legacy_mpipi_frontend.py` module from `finches.frontend` - this was where the original non-inheritance-based frontend had been implemented, but we've now moved to an inheritance-based approach for frontends which has worked without issue, so we're removing legacy entirely. 
	* Added `epsilon_vectors()` to the frontend base to enable direct access to the attractive and repulsive vectors (note this does not involve any sliding window smoothing). 
	* We re-factored the stand-alone stateless epsilon-associated functions previously found in `epsilon_calculations.py` into a new function module called `epsilon_stateless.py`. This avoids a whole host of circular dependency and means various functions can access these broadly-applicable epsilon-associated functions WITHOUT needing to also import the code associated with the Interaction_Matrix_Constructor
	* We removed the automatic code for calculate charge_prefactor - the approach used originally was revised to a new approach which is codified in a jupyter notebook which will be bundled with finches, but this can't be easily automated as it requires some decisions to be made, so we've removed the false sense of being able to automated this.
	* We simplified the import structure so that for most types of analysis a "standard user" might want, you can get there now in a few easy lines:

			from finches import Mpipi_frontend, CALAVDOS_frontend
			mf = Mpipi_frontend()
			cf = CALAVDOS_frontend()
	* Lots of updates to the doc strings to re-write documentation 


* April 8th 2024 - revamped `finches.frontend` module to include CALVADOS and Mpipi frontend classes that can be used for simple access to identical functionality from Mpipi or CALVADOS (or any future model). 

### Copyright

Copyright (c) 2023-2024, Garrett M. Ginell & Alex S. Holehouse

#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.

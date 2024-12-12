FINCHES
==============================
![Finches Logo](finches_logo_v1.png)

### Current version: 0.1.2 (beta public)

## About
FINCHES (First-principle Interactions via CHEmical Specificity) is a software package for computing IDR-associated chemical specificity. It is the accompanying implementation associated with the preprint:

**Holehouse, A. S. Direct prediction of intermolecular interactions driven by disordered regions**
Ginell, G. M., Emenecker, R. J., Lotthammer, J. M., Usher, E. T. & . bioRxiv 2024.06.03.597104 (2024). [doi:10.1101/2024.06.03.597104](http://dx.doi.org/10.1101/2024.06.03.597104) (under review)

### Current status
At this time, we recommend folks use the colab notebooks as their primary route to working with FINCHES. This is primarily because the are still some rough edges, various features are not fully tested, and finches lacks extensive documentation. This will be addressed in the coming weeks but for now our tentative recommendation is to focus on the [colab notebooks linked in the finches-colab repository](https://github.com/idptools/finches-colab).

## Installation
The installation below has been tested in a clean conda environment using Python 3.9; YMMV in other systems. This does not use anything different from our usual stack (`soursop`, `mdtraj`, `numpy`, `scipy` `cython` etc.) so *should* probably install easily into your "default" environment, although I did test on a totally clean conda environment to be safe!

### Create a new conda environment (optional)
If creating a new conda environment, the following command works well (note you can run this anywhere as all the files are created in the conda envs directory):

	conda create -n finches  python=3.9 -y
	
Then activate this environment

	conda activate finches
	
From here we can then install the various dependencies 	
	
### Install dependencies with conda	

First, install `cython`, `numpy`, and `pytorch` using CONDA. 

> NB: This is **very important** because on macOS, we need to ensure we're using consistent Numpy/Pytorch versions from conda OR PyPI, but we cannot mix and match.

Specifically, we recommend the following install instructions (this again can be run anywhere):

	# ensure dependencies are from the same ecosystem (conda)
	conda install numpy pytorch scipy cython matplotlib jupyter  -c pytorch
	
	# we do this separately and last
	conda install mdtraj
	
	# required for integrated disorder prediction; note metapredict is
	# only available via pip
	pip install metapredict 

### Install finches
Next, install finches! This can be done in one of two ways:

#### Install directly from GitHub
The easiest install is to use pip to install directly from GitHub

	pip install git+https://git@github.com/idptools/finches.git
	
We recommend this route unless you expect to make modifications to the code. This can be run anywhere as the installation will place the package in your conda environment's package store location.

#### Install from cloned version
Alternatively, you can clone finches and then install a local working version. 

**Note** This SHOULD be executed in a sensible location because this command will clone (copy) the finches Git repo to a local location:

	git clone git@github.com:idptools/finches.git
		
Once cloned, you can move into the finches directory:

	cd finches

And then install by running the following command inside that directory (i.e. where `pyproject.toml` is):

	pip install -e .
	
Once installed, finches will be available in the environment in which you installed it

**NB**: We use the `-e` flag because it locally links to the code in this directory. This is useful for two reasons:

1. If updates are pushed, you just need to run

	 	git pull
	 	
	 in this directory, and your live version of the code will be updated the next time you use finches. Given that finches are in active development, this is especially useful.
	 
2. It correctly compiles the associated Cython code, and right now, there's something off with the packaging right now, and the Cython code is not correctly distributed on a non `-e` install (Alex to fix!).

### Check the installation worked
To check this has worked, move back to your home directory:

	cd ~

And then run

	python -c  "from finches.frontend.mpipi_frontend import Mpipi_frontend; mf = Mpipi_frontend(); print('Success')"
	
Note this may take a second to run the first time you launch it... 

If this print's "success" then you're good to go! If not something is up...

### Demo
Head on over to `demo/` directory for some Jupyter notebooks showing the types of things you can do with finches.

### Copyright

Copyright (c) 2023-2024, Garrett M. Ginell & Alex S. Holehouse under a CC BY-NC 4.0 license.

### Acknowledgements
Finches was built by Garrett Ginell and Alex Holehouse.


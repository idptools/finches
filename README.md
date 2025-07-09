FINCHES
==============================
![Finches Logo](finches_logo_v1.png)

### Current version: 0.1.3 (beta public)

## About
FINCHES (**F**irst-principle **I**nteractions via **CHE**mical **S**pecificity) is a software package for computing IDR-associated chemical specificity. The FINCHES paper was published in May 2025 [and is available here](https://www.science.org/stoken/author-tokens/ST-2641/full).

### How to use:

* FINCHES is available as a stand-alone Python package (described here)
* FINCHES is available as a set of [Google colab notebooks available here](https://github.com/idptools/finches-colab)
* FINCHES is available as a webserver at [https://www.finches-online.com/](https://www.finches-online.com/)

### Current status of FINCHES software package
FINCHES is currently in a public beta format, which essentially means the code works and runs, but we are actively working on refactoring and restructuring the underlying codebase to improve performance and add new features! As of May 2025 documentation for core FINCHES features is available at [https://finches.readthedocs.io/en/stable/](https://FINCHES.readthedocs.io/en/stable/). 

### Documentation
FINCHES documentation for stable features is available at [https://finches.readthedocs.io/en/stable/](https://finches.readthedocs.io/en/stable/). We encourage you to visit this documentation and raise pull-requests on this GitHub repository if you find issues or errors with the documentation and/or would like additional features documented.

### Usage
Once installed (described below), the recommend use is to interact via the Frontend objects. Both CALVADOS and Mpipi-GG have dedicated Frontend objects which implement a number of useful user-facing functions. Briefly, these frontend objects can be accessed as follows

	# import the modules
	from finches import Mpipi_frontend, CALVADOS_frontend	
	
	# create new instances of the objects; note this 
	# constructor can take several parameters
	mf = Mpipi_frontend()
	
	# create an analagous CALVADOS frontend object (not use here, but 
	# the same functions are usable)
	cf = CALVADOS_frontend()
	
	ddx4_ntd = 'MGDEDWEAEINPHMSSYVPIFEKDRYSGENGDNFNRTPASSSEMDDGPSRRDHFMKSGFASGRNFGNRDAGECNKRDNTSTMGGFGVGKSFGNRGFSNSRFEDGDSSGFWRESSNDCEDNPTRNRGFSKRGGYRDGNNSEASGPYRRGGRGSFRGCRGGFGLGSPNNDLDPDECMQRTGGLFGSRRPVLSGTGNGDTSQSRSGSGSERGGYKGLNEEVITGSGKNSWKSEAEGGES'
	
	# generate a homotypic intermap, note this function 
	# has a large number of parameters that can be passed
	mf.interaction_figure(ddx4_ntd, ddx4_ntd)
	
	# predict and print the homotypic epsilon for this sequence
	print(mf.epsilon(ddx4_ntd, ddx4_ntd))

For more detailed description on how to use [please see our documentation](https://finches.readthedocs.io/en/stable).

### How to cite FINCHES
The core FINCHES publication is:
Ginell, G. M., Emenecker, R. J., Lotthammer, J. M., Keeley, A. T., Plassmeyer, S. P., Razo, N., Usher, E. T., Pelham, J. F. & Holehouse, A. S. Sequence-based prediction of intermolecular interactions driven by disordered regions. Science 388, eadq8381 (2025).

However, if you use FINCHES in you work, please consider citing as follows:

* "*... using FINCHES with the Mpipi-parameters (Mpipi-GG) [1,2,3]* " 

or  

* "*...using FINCHES with the CALVADOS parameters [1,4] "*

**References**

[1] Ginell, G. M., Emenecker, R. J., Lotthammer, J. M., Keeley, A. T., Plassmeyer, S. P., Razo, N., Usher, E. T., Pelham, J. F. & Holehouse, A. S. Sequence-based prediction of intermolecular interactions driven by disordered regions. Science 388, eadq8381 (2025).

[2] Joseph, J. A., Reinhardt, A., Aguirre, A., Chew, P. Y., Russell, K. O., Espinosa, J. R., Garaizar, A. & Collepardo-Guevara, R. Physics-driven coarse-grained model for biomolecular phase separation with near-quantitative accuracy. Nat. Comput. Sci. 1, 732–743 (2021).
 
[3] Lotthammer, J. M., Ginell, G. M., Griffith, D., Emenecker, R. J. & Holehouse, A. S. Direct prediction of intrinsically disordered protein conformational properties from sequence. Nat. Methods 21, 465–476 (2024).
  
[4] Tesei, G., Schulze, T. K., Crehuet, R. & Lindorff-Larsen, K. Accurate model of liquid-liquid phase behavior of intrinsically disordered proteins from optimization of single-chain properties. Proc. Natl. Acad. Sci. U. S. A. 118, e2111696118 (2021).
  

#### Why so many citations?
FINCHES relies on forcefield parameters developed by Joseph et al. & Lotthammer et al, and Tesei et al., as such we strongly encourage folks to cite the papers from which the original forcefields are taken as well as the FINCHES implementation. We note that for the Mpipi parmaters, the defaults used by FINCHES are the Mpipi-GG parameters (developed by Lotthammer et al.), hence the suggestion to cite both the Joseph et al. and the Lotthammer et al. paper. However if this is an issue please ensure the Joseph et al. paper is cited.
  


## Installation
The installation below has been tested in a clean conda environment using Python 3.9+; YMMV in other systems. 

This does not use anything different from our usual stack (`soursop`, `mdtraj`, `numpy`, `scipy` `cython` etc.) so *should* probably install easily into your "default" environment. As of May 2025 we generally recommend Python 3.11 or 3.12 for performance reasons.

### Dependency notes
* **Numpy:** FINCHES will shortly require numpy 2 or higher to run, and we encourage folks to update to a version of numpy above 2.

### FINCHES in conda 
If creating a new conda environment, the following command works well (note you can run this anywhere as all the files are created in the conda envs directory):

	conda create -n finches  python=3.12 -y
	
Then activate this environment

	conda activate finches
	
From here we can then install the various dependencies 

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

### FINCHES in uv
As of July 2025 we (the Holehouse lab) are experimenting with moving from conda to [uv](https://docs.astral.sh/uv/), a new high-performance environment and package manager. We will provide explicit instructions 	for how to set FINCHES up with uv in the 0.1.4 update which may become our recommended installation pipeline going forward.

### Demo
Head on over to `demo/` directory for some Jupyter notebooks showing the types of things you can do with FINCHES. The [documentation](https://finches.readthedocs.io/en/stable/index.html)

### Changelog
Please see [changelog.md](changelog.md) to track version changes.

### Copyright

Copyright (c) 2023-2025, Garrett M. Ginell & Alex S. Holehouse under a CC BY-NC 4.0 license.

### Acknowledgements
FINCHES was built by Garrett Ginell and Alex Holehouse.


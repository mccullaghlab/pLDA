# pLDA

## Overview

This is a package to perform Linear Discriminant Analysis (LDA) on particle positions taking into account the rotational invariance of these positions.  

## Dependencies

This package is dependent on the following packages:

1. Python>=3.6 
2. numpy
3. torch>=1.11 (==1.11 if option 4 is used)
4. sklearn
5. shapeGMMTorch

The examples are also dependent on:

1. MDAnalysis
2. matplotlib

## Installation

After the dependencies have been installed, the package can be installed from pip

`pip install pLDA`

or by downloading from github and then running

`python setup.py install`

## Usage 

This package is designed to mimic the usage of the sklearn package.  You first initiliaze the object and then you can transform additional data.

### Initialize:

`from pLDA import pLDA`

`plda = pLDA.pLDA(training_set_positions, cluster_ids)`

During initialization, the following options are availble:

	- rot_type                - string defining the type of rotational alignment to use.  Options are 'kronecker' and 'uniform'.  Defualt is 'uniform'.
	- kabsch_thresh           - float dictating convergence criteria for each alignment step.  Default value is 1e-1.
	- dtype                   - Torch data type to be used.  Default is torch.float32.
	- device                  - Torch device to be used.  Default is torch.device('cuda:0') device.

### Fit:

`plda.fit(training_set_positions, cluster_ids)`

### Transform:


`transformed_positions = plda.transform(full_trajectory_positions)`

## Attributes

After being properly fit, a pLDA object will have the following attributes:

	- n_clusters		- integer of how many clusters were in tranining cluster_id array
	- n_atoms           	- integer of how many atoms were in the training data
	- n_training_frames    	- integer of how many frames were in the training data
	- lda 			- sklearn LDA object fit using training data
	- lda_vecs              - (n_atoms x 3, n_clusters-1) float array of LD vectors
	- lda_projection        - (n_training_frames, n_clusters-1) float array of LD projections for training data
	- center	      	- (n_atoms, 3) float array of global center/average

Uniform covariance specific attributes

	- var		       	- (n_clusters) float of global variance

Kronecker covariance specific attributes

	- precision	   	- (n_atoms, n_atoms) float array of global precision (inverse covariance)



# Robustness of Meta Matrix Factorization Against Decreasing Privacy Budgets

This repository includes python scripts and ipython-notebooks necessary for conducting experiments utilizing MetaMF and NoMetaMF in the setting of decreasing privacy budgets. Additionally, it provides code for analyzing the user groups in five different datasets available via Zenodo: https://doi.org/10.5281/zenodo.4031011.

## Usage
To reproduce our results, the ipython-notebooks must be executed in following order:

1. Initialize Folder Structure.ipynb: Sets of a hierarchy of folder for saving the experimental results.
2. data/jester/Generation.ipynb: Preprocessing of the Jester dataset utilized in our studies.
3. Identification of User Groups.ipynb: Identification of users with a low, medium or high number of ratings.
4. Train and Evaluate Models.ipynb: Train and evaluate our models (i.e., MetaMF and NoMetaMF) on the datasets provided. 
5. Visualize Results.ipynb: Visualize results of our experiments.
6. Test Personalization and Collaboration.ipynb: Visualize the item embeddings and weights of the rating prediction models.

Furthermore, MetaMF.py includes the implementation of MetaMF and our extension: NoMetaMF. However, MetaMF.py does not need to be run.

## Requirements
* Python 3
* numpy
* pandas
* sklearn
* torch
* matplotlib

## Contributors
* Peter Müllner, Know-Center GmbH, Graz, pmuellner [AT] know [MINUS] center [DOT] at (Contact)
* Dominik Kowald, Know-Center GmbH, Graz
* Elisabeth Lex, Graz University of Technology, Graz
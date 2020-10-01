# Robustness of Meta Matrix Factorization Against Decreasing Privacy Budgets

This repository includes python scripts and ipython-notebooks necessary for conducting experiments utilizing MetaMF and NoMetaMF in the setting of decreasing privacy budgets. The five utilized datasets, i.e., Douban [1], Hetrec-MovieLens [2], MovieLens 1M [3], Ciao [4] and Jester [5] are given within this repository. Additionally, we provide code for analyzing the user groups in these five datasets, which are available via Zenodo: https://doi.org/10.5281/zenodo.4031011.


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

## References
[1] Hu, L., Sun, A., Liu, Y.: Your neighbors affect your ratings: on geographical neighborhood influence to rating prediction. In: SIGIR’14 (2014)
[2] Cantador, I., Brusilovsky, P., Kuflik, T.: Second international workshop on information heterogeneity and fusion in recommender systems (hetrec2011). In: RecSys’11(2011)
[3] Harper, F. M., Konstan, J. A.: The movielens datasets: History and context. ACM Transactions on Interactive Intelligent Systems (TIIS) 5(4), 1–19 (2015)
[4] Guo, G., Zhang, J., Thalmann, D., Yorke-Smith, N.: Etaf: An extended trust antecedents framework for trust prediction. In: ASONAM’14 (2014)
[5] Goldberg, K., Roeder, T., Gupta, D., Perkins, C.:  Eigentaste: A constant time collaborative filtering algorithm. Information Retrieval 4(2), 133–151 (2001)
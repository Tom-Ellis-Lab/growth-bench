# Data for MOMA model (MMANN)

## Input dataset
`complete_dataset.RDS` - It contains gene expression (GE) + metabolic fluxes data (MF)

`gene_expression_dataset.RDS` - It constains GE

**NOTE: To obtain fluxomic data, subtract the data in `complete_dataset.RDS` from `gene_expression_dataset.RDS` (See the example in `run_moma.ipynb` in `models/`)**

## Training and Testing datasets
**NOTE: All data (for training and testing) is included in `complete_dataset.RDS`.**

In order to obtain the testing data, use the the indices corresponding to the datapoints used for test: `indices_for_testing_data.csv`. (See the example in `run_moma.ipynb` in `models/`)

In order to obtain the training data, subtract the testing data from the complete dataset. (See the example in `run_moma.ipynb` in `models/`)

## Weights
`fluxomic_weights.h5` and `gene_expression_weights.h5` - weights for single-view models

**NOTE: The weights for the mult-view model are not included here due their large size. In order to obtain them, please run `run_moma.ipynb` in `models/`. By default the weights are saved in `moma/multi_view_model_GE_MF_0.weights.h5`**

## "Independent" dataset
`independent_dataset.csv` - It contains, GE+MF data for 86 yeast strains: 

- 14 single-gene knockout strains
- 72 double-gene knockout strains (58 of them are imputed via Linear Regression)


## References
https://www.pnas.org/doi/10.1073/pnas.2002959117 



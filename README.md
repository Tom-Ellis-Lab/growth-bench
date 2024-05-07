# Growth Bench
Authors: Piotr Gidzinski and Timon Schneider

A set of tasks comparing machine learning and genome scale metabolic models on the task of predicting growth rates.

## Goal

We want to compare models in their accuracy of predicting growth rates. Thus, we need a standard implementation of different models, a set of datasets combining inputs and targets, and a set of tasks that the models are evaluated on.

### Task 1

- Task: Predict the growth rate in the exponential phase for the S. Cerevisiae.
- Test Dataset: phenotype `growth (exponential phase)` of the yeastphenome.org collection
- Performance Metric: Pearsons Correlation Coefficient

Further Tasks tbd.

## Run the benchmark

To run the benchmark:

1. Create a virtual env to install the bench package with python version 3.10.2 (e.g. with pyenv: `pyenv virtualenv 3.10.2 growth-bench`)
2. Install `bench` package: `pip install -e "bench[all]"`. You can choose to only install dependencies for some of the models using `pip install -e "bench[<model>]"`. (See pyproject.toml for exact names.)
3. Run `python run_benchmark.py` in the root folder.

You can reach out via email to mail@timonschneider.de to get support in case you have troubles running the benchmark.

## Development

This repo contains the python package `bench` that you need to install for development. The softare architecutre roughly follows the strategy pattern (read more [here](https://refactoring.guru/design-patterns/strategy))

### Setup

1. Create a new virtual environment to install dependencies and this package into. The requirements file can be found in `bench/requirements.txt`. Install the requirements using pip: `pip install -r requirements.txt`
2. Install bench as an editable python package: `pip install -e bench`

### How to add a new task

1. Create a new task file in the tasks folder and copy the template from `context.py`.
2. Implement the `benchmark` method that does three things:
   1. Loads the benchmark dataset
   2. Gets a prediction on the benchmark dataset using the strategy object.
   3. Compares pred and true to calculate the performance which is returned.
3. Add predict_taskX methods for each of the models that you would like to evaluate on this task.

### How to add a new model

1. Create a new model file in the `models` folder (follow the naming scheme `[firstAuthorLastName][publicationDate].py`)
2. Follow the implementation in `models/example.py`

# Results

```json
{
  "Task1_RandomNormal": {
    "mse": 0.013432721299113312, 
    "pearson": -0.009558330017799769, 
    "spearman": -0.00140220703592111, 
    "coverage": 1.0
  }, 
  "Task1_SimpleFBA": {
    "mse": 0.022708978466618762, 
    "pearson": 0.03130686259258242, 
    "spearman": 0.015697817879734764, 
    "coverage": 0.1948135447921132
  }, 
  "Task1_Yeast9": {
    "mse": 0.023105975251630537, 
    "pearson": 0.029784631220184526, 
    "spearman": 0.09031747518292652, 
    "coverage": 0.19524217745392203
  }, 
  "Task2_MomaStrategy": {
    "mse": 0.014203120023012161, 
    "pearson": 0.8623281765291522, 
    "spearman": 0.69828412990386, 
    "coverage": 1.0, 
    "r_squared": 0.7379358410835266
  }, 
  "Task2_LassoStrategy": {
    "mse": 0.010653044147426327, 
    "pearson": 0.9016681798226693, 
    "spearman": 0.6644344923579715, 
    "coverage": 1.0, 
    "r_squared": 0.8034388914091549
  }
}
```

## ToDos

- [ ] Search for datasets with growth rates
- [ ] Define interesting tasks to evaluate models on
  - [ ] What are current models predicting => let's define those as tasks
- [ ] Create a standardized way to access data
- [ ] Implement/Port models to work on the prepared datasets
- [ ] Add and evaluate additional models on the datasets

# growth-bench

A set of tasks comparing machine learning and genome scale metabolic models on the task of predicting growth rates.

## Goal

We want to compare models in their accuracy of predicting growth rates. Thus, we need a standard implementation of different models, a set of datasets combining inputs and targets, and a set of tasks that the models are evaluated on.

### Task 1

- Task: Predict the growth rate in the exponential phase for the S. Cerevisiae.
- Test Dataset: phenotype `growth (exponential phase)` of the yeastphenome.org collection
- Performance Metric: Pearsons Correlation Coefficient

Further Tasks tbd.

## Development

This repo contains the python package `bench` that you need to install for development.

### Setup

1. Create a new virtual environment to install dependencies and this package into.
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

## ToDos

- [ ] Search for datasets with growth rates
- [ ] Define interesting tasks to evaluate models on
  - [ ] What are current models predicting => let's define those as tasks
- [ ] Create a standardized way to access data
- [ ] Implement/Port models to work on the prepared datasets
- [ ] Add and evaluate additional models on the datasets

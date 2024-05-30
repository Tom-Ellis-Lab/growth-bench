"""
This file runs the full benchmark.

In the strategy pattern, this is the client.

NOTE: To run multiple strategies, make sure to have all dependencies installed. 
You can find the dependencies for each strategy in the respective model's requirements-{model}.txt file.
"""

from bench.tasks import Task1, Task2
from bench.models import RandomNormal, SimpleFBA
from bench.models import yeast9
from bench.models.moma import moma_strategy
from bench.models.lasso import lasso_strategy

if __name__ == "__main__":
    # The client code picks a concrete strategy and passes it to the context.
    # The client should be aware of the differences between strategies in order
    # to make the right choice.

    tasks = [Task1, Task2]
    models = [
        Woytock2018,
        RandomNormal,
        SimpleFBA,
        yeast9.Yeast9,
        moma_strategy.MomaStrategy,
        lasso_strategy.LassoStrategy,
    ]

    print("\n==== GROWTH BENCH ====\n")

    results = {}

    for task in tasks:
        context = None
        for model in models:
            if context is None:
                context = task(model())
            else:
                context.strategy = model()

            print(f"\nRunning {task.__name__} with {model.__name__}")
            try:
                result = context.benchmark()
                print(f"\tResult: {result}")
                results[f"{task.__name__}_{model.__name__}"] = result
            except NotImplementedError:
                print(f"\t{model.__name__} does not support {task.__name__}")

    # Save benchmark results
    import json

    with open("data/benchmark_results.json", "w") as f:
        json.dump(results, f)

print("\n==== BENCHMARK COMPLETE ====\n")

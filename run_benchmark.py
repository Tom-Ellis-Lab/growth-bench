"""
This file runs the full benchmark.

In the strategy pattern, this is the client.
"""

from bench.tasks import task1, task2, task3, task4
from bench.models import Woytock2018, RandomNormal, SimpleFBA
from bench.models import yeast9_strategy
from bench.models.moma import moma_strategy
from bench.models.lasso import lasso_strategy

if __name__ == "__main__":
    # The client code picks a concrete strategy and passes it to the context.
    # The client should be aware of the differences between strategies in order
    # to make the right choice.

    tasks = [task1.Task1, task2.Task2, task3.Task3, task4.Task4]
    models = [
        Woytock2018,
        RandomNormal,
        SimpleFBA,
        yeast9_strategy.Yeast9Strategy,
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

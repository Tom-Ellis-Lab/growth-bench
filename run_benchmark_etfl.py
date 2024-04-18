"""
This file runs the full benchmark.

In the strategy pattern, this is the client.
"""

from bench.tasks import Task1, Task2
from bench.models.ETFL import cefl

if __name__ == "__main__":
    # The client code picks a concrete strategy and passes it to the context.
    # The client should be aware of the differences between strategies in order
    # to make the right choice.

    tasks = [Task1, Task2]
    models = [cefl.Cefl]

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

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from bigcode_eval.tasks.custom_metrics.multiple_metrics.evaluation import evaluate_problem
from bigcode_eval.tasks.custom_metrics.multiple_metrics.single_experiment_pass_k import for_file

from suite import REVISION, SUITES, load_problems


def passed(results):
    return sum(result["status"] == "OK" and result["exit_code"] == 0 for result in results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    args = parser.parse_args()
    args.out = args.out.resolve()

    problems = load_problems(args.suite)
    generations = json.loads((args.out / "generations.json").read_text())
    if len(generations) != len(problems):
        raise SystemExit(f"{len(generations)} generations for {len(problems)} problems")
    tasks = args.out / "tasks"
    shutil.rmtree(tasks, ignore_errors=True)
    tasks.mkdir()
    for problem, completions in zip(problems, generations):
        path = tasks / f"{problem['name']}.json"
        path.write_text(json.dumps({"name": problem["name"], "language": problem["language"], "prompt": problem["prompt"],
                                    "completions": completions, "tests": problem["tests"]}))
        evaluate_problem(str(tasks), str(path), args.workers)
        path.unlink()

    results = sorted(tasks.glob("*.results.json"))
    if len(results) != len(problems):
        raise SystemExit(f"{len(results)} results for {len(problems)} problems in {tasks}")
    samples = len(generations[0])
    means = np.mean([for_file(path) for path in results], axis=0)
    scores = {
        "suite": f"{args.suite}-rs",
        "revision": REVISION,
        "problems": len(problems),
        "samples": samples,
        **{f"pass@{k}": float(value) for k, value in zip([1, 10, 100], means) if k <= samples},
        "passed": {path.name.removesuffix(".results.json"): passed(json.loads(path.read_text())["results"])
                   for path in results},
    }
    (args.out / "scores.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps({key: value for key, value in scores.items() if key.startswith("pass@")}))


if __name__ == "__main__":
    main()

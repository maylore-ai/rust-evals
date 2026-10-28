import argparse
import json
from pathlib import Path

from suite import SUITES


def cell(scores, key):
    return f"{scores[key] * 100:.1f}" if key in scores else "-"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    scores = {}
    for suite in SUITES:
        for path in sorted(root.glob(f"{suite}/*/scores.json")):
            scores.setdefault(path.parent.name, {})[suite] = json.loads(path.read_text())
    if not scores:
        raise SystemExit(f"no scores.json under {root}")
    header = [f"{suite} pass@{k}" for suite in SUITES for k in (1, 10)]
    print("| model | " + " | ".join(header) + " |")
    print("|---" * (len(header) + 1) + "|")
    for model, suites in sorted(scores.items()):
        cells = [cell(suites.get(suite, {}), f"pass@{k}") for suite in SUITES for k in (1, 10)]
        print(f"| {model} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

from datasets import load_dataset

DATASET = "nuprl/MultiPL-E"
REVISION = "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2"
SUITES = ["humaneval", "mbpp"]


def load_problems(suite):
    return load_dataset(DATASET, f"{suite}-rs", revision=REVISION)["test"]

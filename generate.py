import argparse
import json
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from extract import extract
from suite import SUITES, load_problems

PROMPT = """Complete the following Rust function. Reply with the complete function in a single ```rust code block.

```rust
{prompt}
```"""
ATTEMPTS = 3


def served_models(api):
    with urllib.request.urlopen(f"{api}/models") as response:
        return [model["id"] for model in json.load(response)["data"]]


def complete(api, model, prompt, args):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(prompt=prompt)}],
        "n": args.samples, "temperature": args.temperature, "top_p": 0.95, "max_tokens": args.max_tokens, "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(f"{api}/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                choices = sorted(json.load(response)["choices"], key=lambda choice: choice["index"])
            return [choice["message"]["content"] or "" for choice in choices]
        except urllib.error.HTTPError as error:
            failure = f"HTTP {error.code}: {error.read().decode(errors='replace')[:1000]}"
            if error.code < 500:
                raise RuntimeError(failure) from error
        except (urllib.error.URLError, TimeoutError) as error:
            failure = str(error)
        if attempt == ATTEMPTS:
            raise RuntimeError(f"{ATTEMPTS} attempts failed, last: {failure}")
        print(f"attempt {attempt} failed, retrying: {failure}", flush=True)
        time.sleep(30 * attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api", default="http://localhost:8000/v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--parallel", type=int, default=32)
    args = parser.parse_args()

    problems = load_problems(args.suite)
    names = [problem["name"] for problem in problems]
    prompts = [problem["prompt"].strip() for problem in problems]
    served = served_models(args.api)
    if args.model not in served:
        raise SystemExit(f"{args.api} serves {served}, not {args.model}")
    partial = args.out / "partial"
    partial.mkdir(parents=True, exist_ok=True)

    def one(index):
        path = partial / f"{names[index]}.json"
        if not path.exists():
            replies = complete(args.api, args.model, prompts[index], args)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(replies))
            temporary.rename(path)
        print(f"[{index + 1}/{len(prompts)}]", flush=True)

    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(one, range(len(prompts))))
    replies = [json.loads((partial / f"{name}.json").read_text()) for name in names]
    (args.out / "replies.json").write_text(json.dumps(replies))
    generations = [[extract(prompt, reply) for reply in group] for prompt, group in zip(prompts, replies)]
    (args.out / "generations.json").write_text(json.dumps(generations))
    shutil.rmtree(partial)


if __name__ == "__main__":
    main()

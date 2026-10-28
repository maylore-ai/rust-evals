# rust-evals

English | [Русский](README.ru.md)

Scripts that measure how well a chat model writes Rust on two public benchmarks from [MultiPL-E](https://github.com/nuprl/MultiPL-E).

| suite | dataset config | problems |
|---|---|---|
| HumanEval-Rust | `humaneval-rs` | 156 |
| MBPP-Rust | `mbpp-rs` | 354 |

Problems of both suites are loaded from [`nuprl/MultiPL-E`](https://huggingface.co/datasets/nuprl/MultiPL-E) at revision `28441b6`; [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) at commit `8fc5bae` compiles the programs with `rustc` to run them against the tests. This repository connects them to vLLM through its own chat prompt with a step that pulls the function out of a reply.

Maylore uses these scripts to compare its [Temper](https://huggingface.co/maylore-ai) models with other open models.

## Setup

Needs Linux with an NVIDIA GPU, `uv`, `git`, `curl`, `rustc` and a systemd user session.

```bash
bash setup.sh
```

The script creates `.venv` with Python 3.12 and vLLM 0.30.0 built for CUDA 12.9. The harness is installed into the same environment from a clone in `harness/` at the pinned commit.

`bash setup.sh --score-only` installs CPU PyTorch in place of vLLM. Such an environment scores existing generations without a GPU.

## Running

```bash
bash run.sh -g 0 ~/models/Qwen2.5-Coder-0.5B-Instruct
bash run.sh -s mbpp -g 0,1 -m 0.9 Qwen/Qwen2.5-Coder-7B-Instruct
```

| option | default | meaning |
|---|---|---|
| `-s` | `humaneval,mbpp` | suites to run |
| `-g` | `0` | GPUs for vLLM, several GPUs turn on tensor parallelism |
| `-o` | `results` | output directory |
| `-m` | `0.8` | `--gpu-memory-utilization` for vLLM |
| `-a` | | extra arguments for `vllm serve` |

A model is a local directory or a Hugging Face id whose last path component names the results. `run.sh` refuses two models with the same last component. For every model `run.sh` starts vLLM on port `8000 + <first GPU>`, generates replies for each suite, then stops the server before scoring the replies. A restart skips steps with their output already on disk. Generation also resumes inside a suite from replies saved per problem in `partial/`; a failed request is retried twice. `run.sh` prints the table of all scores in the output directory at the end of its work; `.venv/bin/python summary.py results` prints the same table at any time.

## Output

```
results/
  humaneval/<model>/
    replies.json          raw replies, 50 per problem
    generations.json      programs built from the replies
    scores.json           pass@1, pass@10 and passed samples per problem
    tasks/<problem>.results.json   harness status, exit code, stdout and stderr per sample
  mbpp/<model>/...
  logs/
```

## Method

Every problem goes to the model as one user message.

````
Complete the following Rust function. Reply with the complete function in a single ```rust code block.

```rust
{prompt}
```
````

`{prompt}` is the MultiPL-E prompt with surrounding whitespace stripped. It holds the doc comment and the signature of the function.

Each request asks for 50 samples with temperature 0.2, top_p 0.95, at most 1000 new tokens and seed 0. The request also sets `chat_template_kwargs: {"enable_thinking": false}` to make models with a thinking mode such as Qwen3.5 answer directly. vLLM serves the model in bfloat16 with `--max-model-len 8192` and `--language-model-only`. `VLLM_USE_FLASHINFER_SAMPLER=0` keeps sampling on the PyTorch sampler.

`extract.py` turns a reply into a program.

1. A `<think>` block is removed.
2. The first fenced code block is taken. Only a line starting with `` ``` `` closes a block; a reply without a closed block, such as one cut off by the token limit, is used whole as the code.
3. If the code defines the function from the prompt, its body is kept. Other items from the block (`use`, helper functions, structs) are moved to the start of the body. Doc comments and attributes right above the function are dropped.
4. Without that function the whole code becomes the body. When it has more `}` than `{`, everything from its last `}` on is cut off.
5. The result is the prompt followed by the indented body. The harness appends the MultiPL-E tests. They start with the closing brace of the function.

The harness compiles each program with `rustc` under a 15 second limit, giving the run 5 seconds. A sample passes when the program exits with code 0. pass@k is the unbiased estimator from [Chen et al. 2021](https://arxiv.org/abs/2107.03374) averaged over problems. Scoring runs inside a `systemd-run` scope capped at 12 GiB of RAM with swap disabled, with one worker per CPU thread minus one. Generated programs run on the host with no other isolation.

## Results

Measured on two RTX 3060 12 GB with driver 580, vLLM 0.30.0, PyTorch 2.13.0+cu129 and rustc 1.99.1 by the earlier Temper benchmark scripts with the same prompt, sampling, extraction and scoring. Models from 4B up (Qwen3.5-4B, Qwen3-4B, Qwen2.5-Coder-7B-Instruct, Tessa-Rust-T1-7B, Qwen3.5-9B) ran on both GPUs with tensor parallelism at `-m 0.92`, every other model on one GPU. Qwen3.5-9B also ran with `-a "--max-num-seqs 64 --max-num-batched-tokens 4096"` because vLLM otherwise runs out of GPU memory while capturing CUDA graphs.

| model | HumanEval pass@1 | HumanEval pass@10 | MBPP pass@1 | MBPP pass@10 |
|---|---|---|---|---|
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), thinking off | 76.0 | 90.2 | 65.6 | 79.2 |
| [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 72.4 | 82.6 | 60.5 | 70.0 |
| [Tessa-Rust-T1-7B](https://huggingface.co/Tesslate/Tessa-Rust-T1-7B) | 71.5 | 79.0 | 57.8 | 68.9 |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), thinking off | 66.2 | 81.4 | 57.7 | 73.4 |
| Temper 2 Axon | 64.2 | 73.8 | 54.7 | 61.7 |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), thinking off | 59.8 | 64.8 | 55.6 | 60.6 |
| [Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) | 56.5 | 71.9 | 44.8 | 60.0 |
| [Temper 1 1.5B](https://huggingface.co/maylore-ai/Temper-1-1.5B) | 54.1 | 68.8 | 48.6 | 59.2 |
| [Temper 2 Ion](https://huggingface.co/maylore-ai/Temper-2-Ion) | 48.1 | 57.6 | 45.5 | 53.8 |
| [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | 39.0 | 56.1 | 38.3 | 52.1 |
| [Temper 1 0.5B](https://huggingface.co/maylore-ai/Temper-1-0.5B) | 36.3 | 46.3 | 39.3 | 46.9 |
| [ThoxEdge-RustCoder-0.5B](https://huggingface.co/Thox-ai/ThoxEdge-RustCoder-0.5B) | 28.7 | 44.8 | 33.2 | 43.1 |
| [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) | 23.8 | 39.9 | 28.4 | 40.2 |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), thinking off | 10.1 | 22.1 | 14.5 | 25.5 |

Sampling on a GPU is not bit-exact. Two runs of the same model on this machine differ by about 0.1 point of pass@1. Two fresh `run.sh` runs land within 0.2 point of the table in pass@1 for Qwen2.5-Coder-0.5B-Instruct; Qwen2.5-Coder-1.5B-Instruct run with a smaller batch through `-a "--max-num-seqs 64 --max-num-batched-tokens 4096"` lands within 0.3 point.

The test of MBPP-Rust problem `mbpp_130_max_occurrences` expects one particular element among several with equal frequency; a solution that walks a `HashMap` passes it only with a lucky random hash seed. The harness runs identical programs once, giving repeats the stored result. For a model with one such solution in all 50 samples this means a difference in MBPP pass@1 of up to 0.28 point between two scorings of the same generations.

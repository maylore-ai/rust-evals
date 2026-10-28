# rust-evals

[English](README.md) | Русский

Скрипты, которые измеряют, как чат-модель пишет на Rust, на двух открытых бенчмарках из [MultiPL-E](https://github.com/nuprl/MultiPL-E).

| набор | конфиг датасета | задач |
|---|---|---|
| HumanEval-Rust | `humaneval-rs` | 156 |
| MBPP-Rust | `mbpp-rs` | 354 |

Задачи обоих наборов загружаются из [`nuprl/MultiPL-E`](https://huggingface.co/datasets/nuprl/MultiPL-E) на ревизии `28441b6`, программы через `rustc` с тестами прогоняет [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) на коммите `8fc5bae`. Этот репозиторий соединяет их с vLLM через свой чат-промпт с извлечением функции из ответа.

Этими скриптами Maylore сравнивает свои модели [Temper](https://huggingface.co/maylore-ai) с другими открытыми моделями.

## Установка

Нужны Linux x86_64 с GPU NVIDIA и драйвером под CUDA 12.9, `uv`, `git`, `curl`, `rustc`, доступ к Hugging Face и пользовательская сессия systemd с рабочим `MemoryMax` для `systemd-run --user`.

```bash
bash setup.sh
```

Скрипт создаёт `.venv` с Python 3.12 и vLLM 0.30.0 под CUDA 12.9. Harness ставится в то же окружение из клона в `harness/` на зафиксированном коммите.

`bash setup.sh --score-only` ставит PyTorch для CPU вместо vLLM. Такое окружение оценивает готовые генерации без GPU, потому что `run.sh` поднимает vLLM только для наборов без `generations.json`. Смена режима пересоздаёт `.venv`. `.venv/bin/pytest` запускает тесты `extract.py`.

## Запуск

```bash
bash run.sh -g 0 ~/models/Qwen2.5-Coder-0.5B-Instruct
bash run.sh -s mbpp -g 0,1 -m 0.9 Qwen/Qwen2.5-Coder-7B-Instruct
```

| опция | по умолчанию | значение |
|---|---|---|
| `-s` | `humaneval,mbpp` | наборы для запуска |
| `-g` | `0` | GPU для vLLM в порядке `nvidia-smi`, несколько GPU включают tensor parallelism |
| `-o` | `results` | папка для результатов |
| `-m` | `0.8` | `--gpu-memory-utilization` для vLLM |
| `-a` | | дополнительные аргументы `vllm serve`, делятся по пробелам, можно повторять |

Модель задаётся локальной папкой или id на Hugging Face, результаты называются по последнему компоненту пути. Две модели с одинаковым последним компонентом `run.sh` не принимает. Путь модели записывается в файл `model` папки результатов, на папке с другой моделью `run.sh` останавливается. Для каждой модели `run.sh` запускает vLLM на порту `8000 + <первая GPU>`, генерирует ответы по каждому набору, после остановки сервера оценивает ответы. Перезапуск пропускает шаги с готовым результатом на диске. Генерация внутри набора тоже продолжается с места остановки по ответам, сохранённым по задачам в `partial/`. Запрос с ошибкой HTTP 5xx или обрывом соединения повторяется через 30 секунд, потом через 60. Любая другая ошибка HTTP останавливает генерацию. Таблицу всех оценок из папки результатов `run.sh` печатает в конце работы, ту же таблицу в любой момент выводит `.venv/bin/python summary.py results`.

## Файлы результатов

```
results/
  humaneval/<model>/
    model                 путь или id модели
    replies.json          сырые ответы, 50 на задачу
    generations.json      программы, собранные из ответов
    scores.json           pass@1, pass@10 и прошедшие сэмплы по задачам
    tasks/<problem>.results.json   статус harness, код выхода, stdout и stderr по каждому сэмплу
  mbpp/<model>/...
  logs/                   логи vLLM, генерации и проверки по моделям
```

## Метод

Каждая задача уходит модели одним сообщением пользователя.

````
Complete the following Rust function. Reply with the complete function in a single ```rust code block.

```rust
{prompt}
```
````

`{prompt}` это промпт MultiPL-E без пробельных символов по краям. В нём doc-комментарий и сигнатура функции, в 11 задачах ещё строка `use`.

Каждый запрос просит 50 сэмплов с temperature 0.2, top_p 0.95, не больше 1000 новых токенов и seed 0. `generate.py` держит 32 таких запроса одновременно. Каждый запрос также задаёт `chat_template_kwargs: {"enable_thinking": false}`, чтобы модели с режимом размышлений вроде Qwen3.5 отвечали сразу. vLLM запускает модель в bfloat16 с `--max-model-len 8192` и `--language-model-only`. `VLLM_USE_FLASHINFER_SAMPLER=0` оставляет сэмплирование на сэмплере PyTorch.

`extract.py` превращает ответ в программу.

1. Удаляются все блоки `<think>`. Незакрытый удаляется до конца ответа.
2. Берётся первый блок кода. Закрывает блок только строка, начинающаяся с `` ``` `` после необязательного отступа. Ответ без закрытого блока (например, обрезанный по лимиту токенов) целиком идёт в код.
3. Если код определяет функцию из промпта, остаётся её тело. Остальное содержимое блока (`use`, вспомогательные функции, структуры) переносится в начало тела. Doc-комментарии и атрибуты прямо над функцией отбрасываются.
4. Без этой функции телом становится весь код. Если `}` в нём больше, чем `{`, всё начиная с последней `}` отрезается.
5. Результат состоит из промпта и тела с отступом. Harness дописывает тесты MultiPL-E, они начинаются с закрывающей скобки функции.

Harness компилирует каждую программу `rustc` с лимитом 15 секунд, на запуск даёт 5 секунд. Флагов `rustc` не получает. Программы собираются в редакции 2015 в отладочном режиме с проверкой переполнения целых. Сэмпл проходит, если программа завершилась с кодом 0. pass@k это несмещённая оценка из [Chen et al. 2021](https://arxiv.org/abs/2107.03374), усреднённая по задачам. Оценка идёт внутри scope `systemd-run` с лимитом 12 ГиБ ОЗУ без swap. Число воркеров на единицу меньше числа потоков CPU. Других средств изоляции у сгенерированных программ на хосте нет.

## Результаты

Измерено на двух RTX 3060 12 ГБ с драйвером 580, vLLM 0.30.0, PyTorch 2.13.0+cu129 и rustc 1.99.1 прежними скриптами бенчмарков Temper с тем же промптом, сэмплированием, извлечением функции и проверкой. Модели от 4B (Qwen3.5-4B, Qwen3-4B, Qwen2.5-Coder-7B-Instruct, Tessa-Rust-T1-7B, Qwen3.5-9B) работали на обеих GPU с tensor parallelism при `--gpu-memory-utilization 0.92`, остальные модели на одной. Qwen3.5-9B запускалась с `--max-num-seqs 64 --max-num-batched-tokens 4096`, без них vLLM падает с OOM при захвате CUDA graph.

| модель | HumanEval pass@1 | HumanEval pass@10 | MBPP pass@1 | MBPP pass@10 |
|---|---|---|---|---|
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) без thinking | 76,0 | 90,2 | 65,6 | 79,2 |
| [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 72,4 | 82,6 | 60,5 | 70,0 |
| [Tessa-Rust-T1-7B](https://huggingface.co/Tesslate/Tessa-Rust-T1-7B) | 71,5 | 79,0 | 57,8 | 68,9 |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) без thinking | 66,2 | 81,4 | 57,7 | 73,4 |
| Temper 2 Axon | 64,2 | 73,8 | 54,7 | 61,7 |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) без thinking | 59,8 | 64,8 | 55,6 | 60,6 |
| [Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) | 56,5 | 71,9 | 44,8 | 60,0 |
| [Temper 1 1.5B](https://huggingface.co/maylore-ai/Temper-1-1.5B) | 54,1 | 68,8 | 48,6 | 59,2 |
| [Temper 2 Ion](https://huggingface.co/maylore-ai/Temper-2-Ion) | 48,1 | 57,6 | 45,5 | 53,8 |
| [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | 39,0 | 56,1 | 38,3 | 52,1 |
| [Temper 1 0.5B](https://huggingface.co/maylore-ai/Temper-1-0.5B) | 36,3 | 46,3 | 39,3 | 46,9 |
| [ThoxEdge-RustCoder-0.5B](https://huggingface.co/Thox-ai/ThoxEdge-RustCoder-0.5B) | 28,7 | 44,8 | 33,2 | 43,1 |
| [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) | 23,8 | 39,9 | 28,4 | 40,2 |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) без thinking | 10,1 | 22,1 | 14,5 | 25,5 |

Сэмплирование на GPU не воспроизводится бит в бит. Три новых прогона `run.sh` с Qwen2.5-Coder-0.5B-Instruct дали HumanEval pass@1 23,5, 23,7 и 24,0 при 23,8 в таблице. Прогон Qwen2.5-Coder-1.5B-Instruct с `-a "--max-num-seqs 64 --max-num-batched-tokens 4096"` дал 38,7 при 39,0.

Тест задачи MBPP-Rust `mbpp_130_max_occurrences` ждёт один определённый элемент среди нескольких с равной частотой, решение с обходом `HashMap` проходит его только при удачном случайном seed хеша. Одинаковые программы harness запускает однократно, повторы получают сохранённый результат. Если во всех 50 сэмплах модели одна и та же такая программа, две оценки одних генераций расходятся по MBPP pass@1 до 0,28 пункта.

Вывод программ harness декодирует как строгий UTF-8. Программа с невалидным UTF-8 в выводе роняет `score.py` с `UnicodeDecodeError`.

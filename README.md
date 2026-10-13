# LLM Engineering (`llm` branch: learning notes)

Hands-on experiments with the Anthropic Claude API: basic requests, system prompts, streaming, JSON handling, prompt evaluation and grading, and tool use. Everything lives in a small installable Python package (`llm_engineering`) so scripts share one set of reusable helpers.

> **About this branch.** This branch exists for **learning purposes**. The code has been modified and extended (different structure, helper functions, tooling, notes and workarounds), but the foundation of the code and the concepts it practices come from the course **Building with the Claude API** by **Claude Academy**. It is a personal study project, not an official or production-ready implementation.

## Table of Contents

1. [Repository structure](#1-repository-structure)
2. [Why `pyproject.toml`, `Makefile` and `uv.lock`](#2-why-pyprojecttoml-makefile-and-uvlock)
3. [Getting started](#3-getting-started)
4. [The helper library: `helpers.py`](#4-the-helper-library-helperspy)
5. [NoneType handling in the eval pipeline](#5-nonetype-handling-in-the-eval-pipeline)
6. [Learning scripts](#6-learning-scripts)
7. [Why `stop_sequences` fails for clean JSON](#7-why-stop_sequences-fails-for-clean-json)
8. [`add_assistant_message` and `clean_json`](#8-add_assistant_message-and-clean_json)
9. [Known issues](#9-known-issues)
10. [References and license](#10-references-and-license)

---

## 1. Repository structure

```text
LLM-engineering/
├── .gitignore                  # Ignores .env (API key) and .vscode
├── LICENSE                     # MIT
├── Makefile                    # Shortcuts: setup, sync, lint, format, test, clean
├── README.md
├── pyproject.toml              # Package metadata, dependencies, ruff + pytest config
├── uv.lock                     # Exact, hash-pinned dependency versions (managed by uv)
└── src/
    └── llm_engineering/
        ├── json/
        │   └── dataset.json            # Generated eval dataset (3 tasks)
        ├── learn/                      # Runnable learning scripts
        │   ├── access_claude_api.py
        │   ├── eval_and_grading.py
        │   └── use_tools.py
        ├── schemas/
        │   └── tools/                  # Tool definitions in YAML
        │       ├── add_duration_to_datetime_schema.yaml
        │       └── get_current_datetime_schema.yaml
        └── utils/
            ├── helpers.py              # Reusable functions used by every script
            └── context_conversation.py # Multi-turn context demo
```

`src/llm_engineering.egg-info/` and `__pycache__/` are build and runtime artifacts generated automatically. They are not part of the source.

**Why a `src/` layout?** Code is only importable once the package is installed, so scripts always run against the installed package (`from llm_engineering.utils.helpers import ...`) and not against whatever happens to be in the working directory. That catches packaging mistakes early.

---

## 2. Why `pyproject.toml`, `Makefile` and `uv.lock`

Each file has a different job, and together they make the project easy to set up, consistent and reproducible.

### `pyproject.toml`: one file for the project definition

| Advantage | How it shows up in this repo |
|---|---|
| **Single standard config file** (PEP 621) | Replaces `setup.py`, `setup.cfg`, `requirements.txt` and tool-specific files such as `ruff.toml` and `pytest.ini`. |
| **Declares the package and how to build it** | `[build-system]` uses setuptools; `[tool.setuptools.packages.find] where = ["src"]` points it at the `src/` layout. |
| **Separates runtime from dev dependencies** | Runtime: `anthropic`, `ipykernel`, `pyyaml`. Dev: `ruff`, `pytest`, `pytest-cov`, `python-dotenv` (as an optional extra and a `uv` dependency group). |
| **Central tool configuration** | Ruff (line length 100, Python 3.11 target, lint rules `E F I B UP`) and pytest (`pythonpath = ["src"]`) are configured in the same file. |
| **Enforces a Python version** | `requires-python = ">=3.11"` fails early on unsupported interpreters. |

### `Makefile`: short, memorable commands

A Makefile turns multi-step commands into one-word shortcuts, so the workflow is self-documenting and identical on every machine.

| Command | What it does |
|---|---|
| `make sync` | `python -m uv sync --dev`: installs exactly what `uv.lock` specifies, including dev tools. **Recommended.** |
| `make setup` | Creates `.venv` and runs `pip install -e ".[dev]"`. Alternative that does *not* use `uv.lock`. |
| `make lint` | `ruff check .` |
| `make format` | `ruff format .` and `ruff check --fix .` |
| `make test` | `pytest` |
| `make clean` | Removes `__pycache__`, `.pytest_cache` and `.ruff_cache`. |

### `uv.lock`: reproducible installs

`pyproject.toml` says *what ranges* of packages are acceptable (for example `anthropic>=1.13.0`). `uv.lock` records the *exact versions* that were resolved, for direct **and** transitive dependencies (58 package entries), along with file hashes.

- **Reproducibility:** everyone, and every future you, gets the same versions, so "it worked last month" stays true.
- **Integrity:** hashes confirm that the downloaded files are the ones that were locked.
- **Speed and consistency:** `uv sync` installs straight from the lock file without re-resolving.
- **Safer upgrades:** versions change only when you deliberately run `uv lock --upgrade` and review the diff.

Commit `uv.lock` to git. Do not edit it by hand.

---

## 3. Getting started

1. **Install** (requires Python 3.11+ and [uv](https://docs.astral.sh/uv/)):

   ```bash
   make sync
   ```

   `python-dotenv` is imported by `helpers.py` at runtime but is declared as a dev dependency, so install with the dev group (`make sync` does this).

2. **Add your API key** to a `.env` file in the project root (already git-ignored):

   ```env
   ANTHROPIC_API_KEY=your-key-here
   ```

3. **Run a script** from the project root:

   ```bash
   uv run python -m llm_engineering.learn.access_claude_api
   uv run python -m llm_engineering.learn.eval_and_grading
   uv run python -m llm_engineering.learn.use_tools
   ```

   These scripts call the live API and consume tokens. The model is set once, in `helpers.py` (`model = "claude-haiku-5-5"`).

---

## 4. The helper library: `helpers.py`

`src/llm_engineering/utils/helpers.py` holds everything the scripts share. Functions are grouped by purpose.

### Setup and client

| Name | Description |
|---|---|
| `load_dotenv()`, `API_KEY`, `model` | Module-level setup. Loads `.env`, reads `ANTHROPIC_API_KEY`, and defines the default `model` (currently `claude-haiku-5-5`) used by every script. Change the model in one place. |
| `get_anthropic_client()` | Reads the API key from the environment and returns an `Anthropic` client. |

### Messaging

| Function | Description |
|---|---|
| `add_user_message(messages, message)` | Appends a `user` turn to the history list. Accepts plain text, a list of content blocks (such as tool results), or an SDK `Message` object (in which case its `.content` is used). |
| `add_assistant_message(messages, text)` | Appends an `assistant` turn to the history list. Needed to keep multi-turn context and tool loops working. See [section 8](#8-add_assistant_message-and-clean_json). |
| `chat(messages, model, system=None, stop_sequences=None, tools=None, max_tokens=1000, return_message=False)` | The reusable wrapper around `client.messages.create`. Builds the request from optional arguments, then returns either the first text block (default, with the fallback string `"No text response found"`) or the full `Message` object (`return_message=True`, needed for tool use because `stop_reason` and `tool_use` blocks are required). **`max_tokens` defaults to 1000.** |
| `text_from_message(message)` | Joins every text block of a `Message` into one newline-separated string. |

### JSON cleanup

| Function | Description |
|---|---|
| `clean_json(text)` | Strips whitespace, removes a leading ` ```json ` (or plain ` ``` `) fence and a trailing ` ``` `, then runs `json.loads`. Returns the parsed object, or **`None`** (after printing the error) when the text is not valid JSON. See [section 8](#8-add_assistant_message-and-clean_json) for what it can and cannot handle. |

### Evaluation pipeline

| Function | Description |
|---|---|
| `generate_dataset(model)` | Asks Claude for 3 evaluation tasks (Python, JSON or regex tasks for AWS) and returns them via `clean_json`. |
| `run_prompt(test_case)` | Wraps one task in a "Please solve the following task" prompt and returns Claude's answer. |
| `grade_by_model(test_case, output)` | Model-based grading. Asks Claude to act as a code reviewer and reply with JSON only (`strengths`, `weaknesses`, `reasoning`, `score` from 1 to 10), then parses it with `clean_json`. May return `None`. |
| `validate_json(text)` / `validate_python(text)` / `validate_regex(text)` | Code-based syntax checks using `json.loads`, `ast.parse` and `re.compile`. Each returns `10` if valid and `0` if not. |
| `grade_syntax(output, test_case)` | Picks the right validator from `test_case["format"]` (`"json"`, `"python"` or `"regex"`). Returns `0` when no format is given. |
| `run_test_case(test_case)` | Runs the prompt, grades it, and returns `{output, test_case, score, reasoning}`. Final score = average of the model score and the syntax score. Contains the NoneType handling described in [section 5](#5-nonetype-handling-in-the-eval-pipeline). |
| `run_eval(dataset)` | Runs every test case, averages the scores that exist, prints `Average score: X (n/total graded)`, and returns all results. |

### Tool use

| Function | Description |
|---|---|
| `get_current_datetime(date_format=...)` | Returns the current date and time as a string. Raises `ValueError` if the format is empty. |
| `add_duration_to_datetime(amount, unit, datetime_str=None, date_format=...)` | Adds (or, with a negative amount, subtracts) seconds, minutes, hours, days or weeks. Starts from `datetime_str` if given, otherwise from now. Raises `ValueError` for any other unit. |
| `load_tool_schema(path)` | Loads a tool definition from a YAML file with `yaml.safe_load`. |
| `run_tool(tool_name, tool_input)` | Dispatcher: maps a tool name from Claude to the matching Python function. Raises `ValueError` for unknown tools. |
| `run_tools(message)` | Executes every `tool_use` block in a response and builds the matching `tool_result` dictionaries (`tool_use_id`, `content`, `is_error`). Exceptions are caught and returned with `is_error: True`, so Claude sees the failure instead of the program crashing. |
| `run_conversation(messages, tools)` | The tool loop: call Claude, store its reply with `add_assistant_message`, and if `stop_reason == "tool_use"` run the tools, send the results back, and repeat until Claude gives a final answer. Returns the full history. |
| `get_text_edit_schema(model)` | Returns the text-editor tool definition that matches the model family (`claude-3-7-sonnet` or `claude-3-5-sonnet`); returns `None` for any other model, including the current default. Not used by the current scripts. |

---

## 5. NoneType handling in the eval pipeline

The grader asks Claude to answer in JSON only, and `clean_json` returns `None` whenever that text cannot be parsed. The pipeline handles this in two places.

```python
# run_test_case
model_grade = grade_by_model(test_case, output)
if model_grade is None:
    return {"output": output, "test_case": test_case,
            "score": None, "reasoning": "Grader returned invalid JSON"}

# run_eval
scores = [r["score"] for r in results if r["score"] is not None]
average_score = mean(scores) if scores else 0
print(f"Average score: {average_score} ({len(scores)}/{len(results)} graded)")
```

### Why the handling matters

- **It prevents a crash.** Without the check, `model_grade["score"]` would raise `TypeError: 'NoneType' object is not subscriptable`, and one bad grader response would stop the whole run. `mean()` would also fail if a `None` score were left in the list.
- **It keeps the failure visible.** The test case is still returned, with `score: None` and a clear reason, so it can be inspected and re-run instead of silently disappearing.
- **It avoids a misleading average.** Failed gradings are excluded rather than counted as `0`, so a formatting failure is not mistaken for a bad answer.

### What the output tells you

The `(n/total graded)` suffix is the diagnostic. When `n` is lower than `total`, some grader responses were not valid JSON, and the average covers only the graded subset. A bare `0` is ambiguous: it also appears when *nothing* was graded, because `run_eval` falls back to `0` when the score list is empty. The graded count is what tells those cases apart.

**In this project's run, not all 3 tasks were graded.** The cause is the `max_tokens=1000` limit set in `chat()`, which breaks the grader's JSON so that `clean_json` returns `None`. There are two documented ways a 1000-token cap leads there:

1. **Truncated JSON.** The grader writes `strengths`, `weaknesses`, `reasoning` and `score`. If the reply hits the cap, it is cut off mid-JSON, and `json.loads` fails.
2. **No text block at all.** `claude-haiku-5-5` thinks (adaptive thinking) by default, and thinking tokens count toward `max_tokens`. A long thinking pass can use up the budget before any text is written. `chat()` then returns its fallback string, `"No text response found"`, which is not JSON, so `clean_json` returns `None`.

**How to fix or confirm it:** raise `max_tokens` for grader calls, ask for shorter output, or check `message.stop_reason`. A value of `"max_tokens"` confirms the cap was hit (this needs `return_message=True`, since `chat()` returns only text by default). Checking whether `message.content` contains a text block tells you which of the two cases happened.

---

## 6. Learning scripts

All three live in `src/llm_engineering/learn/` and import the shared helpers.

### `access_claude_api.py`: API fundamentals

Walks through the basics, one scenario after another:

1. **First request.** Creates a client and sends a single message through `client.messages.create` ("Explain CECL in one sentence"), then prints the first content block.
2. **System prompt.** A patient math-tutor system prompt that guides the student step by step instead of giving the answer, shown with the raw SDK call.
3. **Reusable `chat()`.** The same tutor scenario through the helper, to show how much boilerplate it removes.
4. **Streaming.** First the raw event stream (`stream=True`, with the loop commented out), then the SDK's simpler `client.messages.stream(...)` interface with `text_stream` and `get_final_message()`.
5. **JSON-only responses.** Requests an EventBridge rule as JSON only, then post-processes it with `clean_json`. The module docstring explains why the stop-sequence approach is not used (see [section 7](#7-why-stop_sequences-fails-for-clean-json)).

### `eval_and_grading.py`: dataset generation and prompt evaluation

An end-to-end evaluation loop:

1. `generate_dataset` asks Claude for 3 tasks (Python, JSON or regex, AWS-related).
2. The tasks are saved to `src/llm_engineering/json/dataset.json`, then loaded back from disk.
3. `run_eval` runs every task through Claude, grades each answer (model grade plus syntax grade), and prints the average with the graded count.
4. The full results list is printed.

Because every grader call depends on `clean_json`, this is the script where the [NoneType handling](#5-nonetype-handling-in-the-eval-pipeline) matters.

### `use_tools.py`: tool use

1. **Loads tool definitions** from YAML (`get_current_datetime_schema.yaml`, `add_duration_to_datetime_schema.yaml`) and wraps them in `ToolParam`. This is a typing aid for static checkers such as pyright or mypy. It does not validate the schema at runtime.
2. **Runs the manual tool round trip** for "What is the exact time, formatted as HH:MM:SS?": send the question with `tools=[...]`, receive a `tool_use` block, run `get_current_datetime`, send back a `tool_result`, and print Claude's final answer. Tools exist because a model cannot know real-time information, such as the current time, by itself. Claude's reply is stored by echoing `response.content` back unchanged.
3. **Runs a multi-tool conversation.** "Take current date and add 103 days to it" needs two tools in sequence: get the current date, then add the duration. `run_conversation` automates the loop, and the script prints the final text.

The utility script `utils/context_conversation.py` is a small multi-turn demo: ask a question, then a follow-up while keeping the history so the model has context.

---

## 7. Why `stop_sequences` fails for clean JSON

### The technique

The course pattern for getting JSON with no extra commentary combines two features:

```python
# Illustrative only. This is NOT in the repo, and it does not work on current models.
add_user_message(messages, "Generate a very short event bridge rule as JSON only")
add_assistant_message(messages, "```json")                   # 1. prefill the opening fence
text = chat(messages, model=model, stop_sequences=["```"])   # 2. stop at the closing fence
```

The prefilled opening fence makes Claude continue with raw JSON, and the stop sequence cuts generation at the closing fence, so the result needs no cleanup.

### What actually fails

`stop_sequences` is a normal, supported API parameter, and `chat()` already forwards it. When Claude emits one of the strings, generation stops, `stop_reason` is `"stop_sequence"`, and the `stop_sequence` field says which one fired. The failure is the **assistant prefill** that the technique depends on:

- **Prefill is rejected on current models.** Per Anthropic's documentation, Claude 4.6 and later models (and Claude Mythos Preview) do not support assistant message prefill. This repo's model, `claude-haiku-5-5`, is in that group. A request whose last message has the `assistant` role returns an HTTP 400 `invalid_request_error` saying the model does not support prefill and the conversation must end with a user message. So the pattern fails at step 1, before `stop_sequences` is ever used.
- **Without the prefill, the stop sequence fires in the wrong place.** Claude starts its own reply, usually with an opening ` ```json ` fence or a sentence of preamble. A stop sequence of ` ``` ` would match that *opening* fence and return an empty or incomplete answer, not clean JSON.
- **Prefill and JSON outputs are mutually exclusive.** The structured outputs documentation lists message prefilling as incompatible with JSON outputs.

So prefill and `stop_sequences` only work as a pair, and the pair is unavailable here. This is why `clean_json` exists. The `stop_sequences` parameter itself is not the culprit and still works for other uses, such as stopping at a custom marker like `END`.

---

## 8. `add_assistant_message` and `clean_json`

### What `add_assistant_message` is for

Claude's API is stateless: every request must carry the whole conversation. `add_assistant_message(messages, text)` appends an `assistant` turn to that history. It serves two different purposes:

| Purpose | Status | Where it matters |
|---|---|---|
| **1. Keep conversation history** (store Claude's reply so the next request includes it) | **Required.** Cannot be replaced. | `run_conversation` calls it on every loop iteration. `use_tools.py` does the same inline with `messages.append`. `context_conversation.py` needs it for its follow-up question (see [Known issues](#9-known-issues)). |
| **2. Prefill** (end the history with a partial assistant turn to steer the output, such as forcing JSON) | **Not supported** on Claude 4.6 and later models, including this repo's `claude-haiku-5-5`. | The commented-out course approach in [section 7](#7-why-stop_sequences-fails-for-clean-json). |

Two practical rules when storing a reply:

- **Pass `response.content` back unchanged.** It is a list of blocks (text, `tool_use`, and possibly thinking blocks). Newer models think by default, and the API returns a 400 error if thinking blocks in the latest assistant turn are edited, reordered or filtered out. Never rebuild the turn from only the text.
- **Do not end a request with an assistant message** on current models. After `add_assistant_message`, the next message must be a user message (a follow-up question or `tool_result` blocks).

### Can `clean_json` replace it?

**For purpose 2 (forcing clean JSON): yes, and it is what this repo does.** Skipping the prefill and cleaning afterwards removes the need to end a request with an assistant message. **For purpose 1 (conversation history): no.** `clean_json` only parses text. It cannot give Claude memory of earlier turns.

`clean_json` was tested against typical outputs:

| Model output | Result |
|---|---|
| Bare JSON | Parsed |
| JSON wrapped in ` ```json ` or ` ``` ` fences, with extra whitespace | Parsed |
| A sentence of preamble before the fence (`Here is the JSON: ...`) | `None` |
| Text after the closing fence (`Hope this helps!`) | `None` |
| JSON cut off by `max_tokens` | `None` |
| The `"No text response found"` fallback from `chat()` | `None` |

It handles the common case (fences only) but is not a guarantee. It also returns `None` instead of raising, so callers must check for it, which is what [section 5](#5-nonetype-handling-in-the-eval-pipeline) does for the grader.

### Making JSON output more reliable

| Option | Notes |
|---|---|
| **Instruct in the prompt or system prompt** ("Respond with only a JSON object, no other text") | Already used by `grade_by_model`. Cheap and works with `clean_json`. Not a guarantee. |
| **Check `stop_reason` before parsing** | If it is `"max_tokens"`, retry with a higher limit instead of parsing broken text. |
| **Structured outputs** (`output_config.format`) | Anthropic's recommended replacement for prefill. Constrains decoding to a JSON schema, so the text is valid JSON that matches the schema. Haiku 5.5 is a supported model. Not currently used in this repo. |

A sketch of structured outputs (illustrative; `chat()` would need an extra `output_config` argument to pass it through):

```python
response = client.messages.create(
    model=model,
    max_tokens=2000,
    messages=[{"role": "user", "content": eval_prompt}],
    output_config={"format": {"type": "json_schema", "schema": {
        "type": "object",
        "properties": {
            "strengths": {"type": "array", "items": {"type": "string"}},
            "weaknesses": {"type": "array", "items": {"type": "string"}},
            "reasoning": {"type": "string"},
            "score": {"type": "integer"},
        },
        "required": ["strengths", "weaknesses", "reasoning", "score"],
        "additionalProperties": False,
    }}},
)
text = next(b.text for b in response.content if b.type == "text")
grade = json.loads(text)
```

**Structured outputs do not fix truncation.** If `stop_reason` is `"max_tokens"`, the output can still be incomplete and fail to match the schema, so the `max_tokens` limit from [section 5](#5-nonetype-handling-in-the-eval-pipeline) still needs attention. Keep `clean_json` as a safety net for the prompt-only path.

---

## 9. Known issues

These are the open items found while documenting the code. Delete this section once they are fixed.

- **`context_conversation.py` is broken.** It calls `chat(messages)` without the required `model` argument (`TypeError`), and its `add_assistant_message(messages, answer)` call is commented out, so the follow-up question would not include Claude's first answer.
- **Syntax grading is always `0` for the generated dataset.** `grade_syntax` reads `test_case["format"]`, but `generate_dataset` only asks for a `task` field, so every final score is the model score divided by 2 (a maximum of 5). Add a `format` field to the generated tasks to fix this.
- **`generate_dataset` can also return `None`** for the same reasons as the grader. `eval_and_grading.py` does not guard that call, so it would write `null` to `dataset.json` and then fail inside `run_eval`.
- **Reading `content[0].text` is fragile on thinking models.** `access_claude_api.py` and `use_tools.py` assume the first content block is text. Newer models can return a thinking block first, which has no `.text`. The existing `text_from_message` helper (or `next(b.text for b in content if b.type == "text")`) is safer.
- **Stale docstring.** The module docstring in `access_claude_api.py` says `add_assistant_message` is commented out in `helpers.py`. It is now active, and the stop-sequence explanation there should point to [section 7](#7-why-stop_sequences-fails-for-clean-json).
- **`pyproject.toml` details.** The pytest `testpaths` (`common/tests`, `phase2_flagship`, `phase3_differentiator`) do not exist in this repo, and `python-dotenv` is a dev-only dependency that the runtime code imports.

---

## 10. References and license

- Claude API errors, including "prefill not supported": <https://platform.claude.com/docs/en/api/errors>
- Structured outputs: <https://platform.claude.com/docs/en/build-with-claude/structured-outputs>
- Troubleshooting thinking (thinking defaults, `max_tokens` stops): <https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting>
- Stop reasons: <https://platform.claude.com/docs/build-with-claude/handling-stop-reasons>
- uv documentation: <https://docs.astral.sh/uv/>
- Course basis: *Building with the Claude API*, Claude Academy

Released under the [MIT License](LICENSE).
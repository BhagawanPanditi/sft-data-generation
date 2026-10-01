# Evidence-grounded coding question-bank pipeline

This repository builds a **prompt-only coding question bank** from coding benchmark failure concepts:

1. `download_datasets.py` — download immutable dataset revisions and safely decode LBPP.
2. `stage1_extract.py` — extract quote-grounded invariants and concrete failure mechanisms.
3. `stage1_verify.py` — hard-gate integrity, grounding, leakage, and distribution quality.
4. `stage2_taxonomy.py` — build and semantically audit a weighted reasoning-skill taxonomy.
5. `stage3_question_bank.py` — generate, adversarially review, decontaminate, deduplicate, and quota-balance 20,000 prompts.

All LLM stages use a global concurrency of **300**. Override endpoints with `VLLM_SERVERS` and use a different reviewer model with `JUDGE_VLLM_SERVERS` / `JUDGE_MODEL_ID`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

```bash
python download_datasets.py
```

The downloader resolves an immutable revision of
[`livecodebench/code_generation_lite`](https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/main),
downloads all six release files, and maps them as follows:

```text
test.jsonl  -> data/coding/livecodebench_v1.jsonl
test2.jsonl -> data/coding/livecodebench_v2.jsonl
...
test6.jsonl -> data/coding/livecodebench_v6.jsonl
```

These files total roughly 4.5 GB. The downloader uses the Hugging Face cache and normally hard-links the cached files rather than making another full copy. It validates every JSONL record and records the pinned revision, row count, local path, and SHA-256 hash in the download manifest. Stage 1 does not assume that releases are disjoint: it deduplicates repeated LiveCodeBench statements and writes the decisions to `output/autopsies/dedup_report.jsonl`.

## Optional: mine actual target-model failures

Plausible failure modes are useful, but execution-observed failures are better. If you have benchmark attempts from the model you intend to improve, write:

```text
data/target_model_failures.jsonl
```

with rows such as:

```json
{"sample_id":"humanevalplus:HumanEval/0","candidate":"...failed code...","failure_observation":"Fails the boundary test because ...","failed_tests":"optional execution log"}
```

Use the retained `sample_id` values printed by stage 1/dedup reporting. Stage 1 adds these as typed `target_attempt` and `target_failure` evidence and instructs the analyst to prefer actual failures. The file is included in the resume fingerprint. Never execute untrusted candidate code outside a sandbox.

## Run

Start OpenAI-compatible vLLM servers, then run:

```bash
python stage1_extract.py
python stage1_verify.py
python stage2_taxonomy.py
python stage3_question_bank.py
```

Later stages require the previous stage's `_SUCCESS` marker. A stage refuses to resume output made with a different input/prompt/config fingerprint.

Useful environment settings:

```bash
# These direct cluster addresses are also the built-in defaults; no SSH port forwarding is needed.
export VLLM_SERVERS=http://cn23-a40:8000/v1,http://cn24-a40:8001/v1,http://cn24-a40:8002/v1
# SERVED_MODEL_ID is optional; omit it to use the model reported by /v1/models.
# export SERVED_MODEL_ID=generator-model-id
# By default reviewers use the same three endpoints. If another cluster host serves a
# genuinely independent reviewer, configure it and enable the strict independence gate:
# export JUDGE_VLLM_SERVERS=http://reviewer-host:8000/v1
# export JUDGE_MODEL_ID=independent-reviewer-model-id
# export REQUIRE_DISTINCT_JUDGE_MODEL=1
# Retry unresolved stage-1 records after inspecting the verifier report.
# export RETRY_REJECTED=1
# If a non-Qwen chat template rejects enable_thinking:
export EXTRA_REQUEST_BODY='{}'
```

The default request body disables Qwen-style hidden thinking so constrained output budgets are spent on JSON rather than reasoning tokens.

## Final artifacts

Public prompt-only bank:

```text
output/question_bank/sft_prompts.jsonl
output/question_bank/sft_question_bank.jsonl  # same public schema, compatibility name
```

Each public row contains only:

```json
{"id":"qb-000000","prompt":"...","language":"python","style":"function"}
```

Private lineage and quality metadata is stored separately:

```text
output/question_bank/sft_prompt_metadata.jsonl
output/question_bank/question_bank_report.json
```

Do **not** serialize the private metadata into training prompts. It includes targeted skills, hidden design contracts, mutation witnesses, and reviewer decisions.

## Why this version is stricter

- LBPP decodes `completion`, `test_setup`, `test_list`, and `test_file` through a restricted unpickler and final JSON layer. Failed records are rejected.
- Statement, solution, test, and trace sections are typed. Evidence must occur in the exact claimed section.
- HumanEval statement and reference solution are separate sections. CRUXEval program/input/output are trace evidence—not tests or a reference solution.
- HumanEval/HumanEval+ and MBPP/MBPP+ use richer-source precedence; exact cumulative-shard duplicates are removed by normalized statement hash.
- `autopsies.jsonl` is canonical; clean and flat views are atomically rebuilt, preventing partial multi-file resume corruption.
- Stage 1 verification combines deterministic checks with a stratified LLM audit of faithfulness, transferability, evidence support, basis labels, invariants, and concrete failures.
- Emergency taxonomy summaries are never cached. Tree checkpoints include code, model, input, and summary fingerprints. Cluster proposals see summary multiplicity, ancestor context, and retained descriptions.
- Terminal size and branching factor are separate. A semantic judge audits sampled concept-to-path assignments.
- Question generation starts with hidden contract and per-skill mutation-witness design, then writes public prose. Local AST checks require synchronous, fully typed, non-executable declarations with safe defaults.
- Descriptive stdin input/output-format headings are allowed; concrete worked input/output values are not.
- Every question must pass two complementary fail-closed reviews: public ambiguity/oracle analysis and hidden-contract/mutation consistency. A separate reviewer model is supported and can be required.
- Cell quotas are explicitly proportional to `cell_size ** 0.5`; final round-robin selection no longer silently overrides the configured distribution.
- Generation continues in persisted rounds until the post-dedup target is reached or the hard round budget is exhausted.
- Composition uses sibling/cousin taxonomy cells and avoids two concepts from the same source task.

## Answer generation and executable calibration

This repository intentionally stops at prompts because answers will be generated elsewhere. Before SFT, the answer pipeline should produce, sandbox, and retain:

1. a reference implementation;
2. tests derived from the full written contract;
3. one mutant per `skill_witness`, implementing its target failure mechanism;
4. at least one valid test where every mutant fails and the reference passes;
5. several independent candidate solutions for empirical pass-rate calibration.

Recommended acceptance criteria:

- reference passes every generated and property-based test;
- each targeted mutant fails a dedicated witness test;
- independent reference solvers agree on deterministic outputs;
- no flaky, environment-dependent, or timeout-sensitive behavior;
- an empirical difficulty/pass-rate band appropriate for the intended training curriculum.

A strong question-only pipeline cannot prove program correctness without executable answers and tests. The private metadata is designed to make that later mutation-testing step straightforward.

## Benchmark-score expectations and evaluation validity

The bank is explicitly targeted at invariants and failure modes mined from HumanEval, MBPP, EvalPlus, EvoEval, ClassEval, LiveCodeBench, CRUXEval, and LBPP. This should increase the chance that subsequent verified answer training improves related capabilities and pass@k, but **no data pipeline can guarantee a score increase**; the result must be established by controlled ablations.

Because these prompts descend from the named benchmarks, improvements measured on those same benchmark items are not a clean held-out generalization claim, even when wording is lexically decontaminated. Preserve `concept_ids`, `source_families`, and source hashes in the private sidecar and report both:

- targeted benchmark improvement; and
- performance on a genuinely held-out, non-descendant coding evaluation.

For pass@k comparisons, keep model, decoding temperature, sample count, evaluator, and test-suite revision fixed, and report confidence intervals across task-level outcomes.

## Tests

```bash
pytest -q
python -m compileall -q .
```

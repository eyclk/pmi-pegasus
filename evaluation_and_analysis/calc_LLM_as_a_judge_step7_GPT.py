"""
STEP 7 -- LLM-as-a-judge with a PAID API MODEL, against the REFERENCE
SUMMARIES.

This is step 5 with the judge swapped out: instead of running Prometheus 7B
locally, it asks an OpenAI model over the API. Everything that defines the
comparison is carried over from calc_LLM_as_a_judge_step5_Prometheus.py
unchanged -- the prompt (verbatim, including the "the source document is not
shown to you" instruction and the note on extra details), the [RESULT] parsing
and the A/B swap.

There is deliberately NO per-dataset prompt here. Steps 6/7 needed one because
they judged against the source document, where a how-to text and a news article
call for different criteria; the reference summary is short, self-contained
ground truth, so all three datasets get the single step 5 prompt and the three
numbers stay directly comparable with each other.

One difference from step 5 survives: the swap is seeded per sample
(swap_for_sample) rather than drawn from the global RNG, so a resumed run is
identical to an uninterrupted one.

Input/output handling: candidates are read from the
"eval_generated_pred/eval_results_*" folders, results are written as one JSON
plus one ".log" per comparison, a per-dataset summary log is rebuilt from disk,
and a ".partial.jsonl" makes the job resumable.

----------------------------------------------------------------------------
COST -- READ THIS BEFORE THE FIRST RUN
----------------------------------------------------------------------------
The full grid is 3 datasets x 8 checkpoints = 24 comparisons over 28401 test
documents each time, i.e. 227208 paid API calls. On gpt-5.6-luna at effort=low
that is ~139M input + ~51M output tokens, about $89 (--estimate, 2026-08-21).

Judging against the reference summary rather than the source document shrinks
the ground-truth text 13x -- 174M tokens of document across the grid against
13M tokens of reference -- but the bill only drops by about a quarter, from
~$121 to ~$89. Two reasons, both worth knowing before trying to optimise
further:

  * the fixed prompt boilerplate is ~450 tokens and is paid 227208 times, i.e.
    ~102M tokens, which is now nearly three quarters of the input side. It sits
    below OpenAI's ~1024-token prefix-caching threshold, so the "cached_input"
    rate in PRICING does not apply to it;
  * output dominates the cost regardless: ~51M tokens at 6x the input rate is
    ~$61 of the ~$89. EXPECTED_OUTPUT_TOKENS, not the prompt length, is the
    number to re-measure when anything changes.

Two levers:

  * --estimate prints the projected token count and cost for the selected
    comparisons and exits WITHOUT calling the API. Always run it first.
  * --max-samples N judges a deterministic random subset instead of the whole
    test set. The subset is seeded per DATASET (not per checkpoint), so all 8
    checkpoints of a dataset are judged on exactly the same documents. NOTE
    that a subset run is NOT directly comparable with the step 5 numbers, which
    were computed over the full test set -- the default here is the full set for
    that reason.

----------------------------------------------------------------------------
DETERMINISM -- WHAT IS AND IS NOT POSSIBLE HERE
----------------------------------------------------------------------------
Step 5/6 can be made bit-for-bit reproducible. This script CANNOT, and no
setting will change that: the GPT-5 series reasoning models reject `temperature`
(fixed at 1) and do not accept `seed`, so the sampling is out of our hands.

What is pinned instead:

  * the exact model id, never the floating alias -- `gpt-5.6` routes to Sol,
    which is 25x the price of Luna and can be repointed by OpenAI at any time;
  * the reasoning effort, which changes both the verdicts and the bill;
  * the prompt (step 5's, verbatim) and the per-sample A/B swap;
  * the results themselves: a sample that has been judged once is never asked
    again, because the ".partial.jsonl" is the cache. Re-running a finished
    comparison costs nothing and returns the identical file.

Each entry records the model, the reasoning effort, the response id and the
token usage, so any drift can be traced afterwards. Because per-sample verdicts
are genuinely stochastic here, treat the aggregate as the measurement and give
it enough samples to be worth reading (see the note on MAX_SAMPLES_PER_COMPARISON).

----------------------------------------------------------------------------
SETUP
----------------------------------------------------------------------------
  1. pip install openai
  2. Create an API key at https://platform.openai.com/api-keys and add a
     payment method under Settings -> Billing. NOTE: a ChatGPT Plus/Pro
     subscription does NOT include API access -- it is billed separately.
  3. export OPENAI_API_KEY="sk-..."      (never hard-code it in this file)
  4. python calc_LLM_as_a_judge_step7_GPT.py --estimate
  5. python calc_LLM_as_a_judge_step7_GPT.py --datasets xsum --checkpoints 8M
"""

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

# Imported softly so that --estimate (which sends nothing) still runs on a
# machine that has not installed the SDK yet. The hard failure happens in
# load_client_if_needed(), i.e. only when a paid request is about to be made.
try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None

###############################################################################
# CONFIGURATION
###############################################################################

# Pinned to the exact model, deliberately NOT the "gpt-5.6" alias: that alias
# routes to Sol at $5.00/$30.00 per 1M tokens, 25x Luna's price, and OpenAI can
# repoint it without notice -- which would silently change both the bill and the
# verdicts halfway through a grid.
#
# Luna is the cost-efficient choice for this job by a wide margin. Judging one
# summary pair against a reference summary is a bounded comparison task, not a
# frontier-reasoning problem, and the whole grid on Sol would cost roughly 25x
# what it costs on Luna. Override with --model if you want to spot-check a
# subset against a stronger judge.
MODEL = "gpt-5.6-luna"

# USD per 1M tokens, standard (non-batch) tier. Checked against
# https://developers.openai.com/api/docs/pricing on 2026-08-18 -- re-check
# before relying on an estimate, these move.
PRICING = {
    # --- budget tier: cheaper than Luna, but weaker judges ------------------
    "gpt-5-nano":    {"input": 0.05, "output": 0.40},
    "gpt-4.1-nano":  {"input": 0.10, "output": 0.40},
    "gpt-4o-mini":   {"input": 0.15, "output": 0.60},
    "gpt-5-mini":    {"input": 0.25, "output": 2.00},
    "gpt-4.1-mini":  {"input": 0.40, "output": 1.60},
    # --- current generation -------------------------------------------------
    "gpt-5.4-nano":  {"input": 0.20, "output": 1.25},
    "gpt-5.6-luna":  {"input": 0.20, "output": 1.20, "cached_input": 0.02},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00, "cached_input": 0.20},
    "gpt-5.6-sol":   {"input": 5.00, "output": 30.00, "cached_input": 0.50},
    "gpt-5.5":       {"input": 5.00, "output": 30.00, "cached_input": 0.50},
}

# The two model families take different knobs, and passing the wrong one is a
# 400, not a warning:
#
#   * GPT-5 series (and the o-series) are REASONING models. They accept
#     reasoning.effort and reject temperature/seed -- temperature is pinned at 1.
#   * GPT-4 series are not. They accept temperature and seed, and reject
#     reasoning. Using one of them buys back much of the determinism this
#     project cares about: temperature=0 plus a fixed seed is not a guarantee
#     (OpenAI documents seed as best-effort, with system_fingerprint telling you
#     when the backend moved underneath you), but it is far steadier than a
#     model whose temperature is stuck at 1.
#
# request_parameters_for() below picks the right set, so --model can be pointed
# at either family without editing anything.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

# Only sent to models that accept it (see above).
SEED = 0

# "none" keeps the model from spending (billed) reasoning tokens before it
# answers. The judging prompt already spells out the procedure step by step, so
# there is little for a reasoning pass to add -- and reasoning tokens are billed
# at the OUTPUT rate, which is 6x the input rate. Raise it to "low"/"medium" if
# a spot-check shows the verdicts improving enough to justify the cost.
REASONING_EFFORT = "low"

# Feedback plus the [RESULT] tag. Step 5 used 512 for Prometheus and saw a
# ~0.2% (cnn) to ~1.0-1.4% (wikihow) rate of generations that never emitted the
# tag; 768 is kept here, unchanged from the source-document version of this
# script, so a missing tag means the judge chose not to emit one rather than ran
# out of room. Note this caps reasoning tokens too when the effort is raised.
MAX_OUTPUT_TOKENS = 768

# Parallel in-flight requests. Unlike step 5's batch size this is purely a
# throughput knob: each request is judged independently by the API, so
# concurrency cannot change a verdict. Lower it if you hit rate limits.
CONCURRENCY = 8

# ---------------------------------------------------------------------------
# HOW MANY SAMPLES -- the single biggest cost decision in this file.
# ---------------------------------------------------------------------------
# None judges the whole test set, which is what steps 4/5/6 do. Keeping that as
# the default is deliberate: a step 7 number is only directly comparable with
# the step 5 number for the same comparison if both saw the same documents, and
# a default that had to be overridden every time would eventually be forgotten,
# producing a subset run that looks complete in the output files.
#
# The cost of that choice is real -- 28401 documents x 8 checkpoints = 227208
# paid calls -- so --estimate exists to price a run before it happens, and
# --max-samples N buys a cheaper, still-usable answer when a full run is not
# worth it. The uncertainty on the PMI-minus-ROUGE gap is roughly
# sqrt((p_pmi + p_rouge) / N), so with the ~41%/37% split step 5 reports on
# wikihow:
#
#     N=  500 per comparison  ->  +/- 3.9 points   (too coarse: the gap is 1-4)
#     N= 2000 per comparison  ->  +/- 2.0 points   (resolves it, ~1/3 the cost
#                                                   on wikihow, ~1/6 on cnn)
#     N= 5000 per comparison  ->  +/- 1.2 points
#     N=full  (5577 wikihow)  ->  +/- 1.2 points   <- the default
#     N=full (11490 cnn)      ->  +/- 0.8 points
MAX_SAMPLES_PER_COMPARISON = None

# Output tokens per verdict, for the --estimate projection only.
#
# MEASURED, not assumed: a 20-sample wikihow/1M run on gpt-5.6-luna at
# effort="none" averaged 120 output tokens (feedback of ~600 characters plus the
# [RESULT] tag), with zero reasoning tokens. The first version of this estimate
# used MAX_OUTPUT_TOKENS // 2 = 384 and overstated the bill by ~3x, because the
# 768 cap is a ceiling the judge never approaches, not a target.
#
# Re-measure if the model, the reasoning effort or the prompt changes -- any
# completed comparison reports its real usage in the .log. Reasoning tokens are
# billed at the OUTPUT rate, so raising the effort above "none" can multiply
# this number several times over; --estimate warns when the two disagree.
# Measured on 20-sample wikihow/1M runs.
#
# Under the OLD source-document prompt:
#   effort="none" -> 126 output tokens (0 reasoning)
#   effort="low"  -> 225 output tokens (114 of them reasoning, billed at the
#                    output rate) -- 1.8x, i.e. the visible feedback got a
#                    little SHORTER while the model spent the budget thinking.
#
# Under the CURRENT step 5 reference-summary prompt, effort="low" (2026-08-21):
#   output    = 209 tokens (mean; median 201, range 103-378)
#   reasoning = 117 of those, leaving ~92 tokens of visible feedback
#   input     = 536 tokens (mean; the char/4 estimator says ~611, i.e. ~14% high)
#
# Nothing came within range of the 768 cap (max 378) and all 20 emitted a
# [RESULT] tag, so MAX_OUTPUT_TOKENS is not binding at this effort.
EXPECTED_OUTPUT_TOKENS = 209
EXPECTED_OUTPUT_TOKENS_MEASURED_AT_EFFORT = "low"

# Retry policy for 429s and transient 5xx. Sleeps 2, 4, 8, 16, 32 seconds.
MAX_RETRIES = 5

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

FINETUNE_DATA_DIR = REPO_ROOT / "finetune_data"
GENERATED_PRED_DIR = REPO_ROOT / "eval_generated_pred"

# Checkpoints (pretraining steps) to compare.
CHECKPOINTS = [f"{i}M" for i in range(1, 9)]

# dataset key -> where its reference summaries / candidate summaries / outputs
# live. No "prompt_style" any more: every dataset is judged with the single
# step 5 prompt (see JUDGE PROMPT below).
DATASETS = {
    "cnn": {
        "finetune_data_folder": "cnn_dailymail_comb",
        "eval_folder_suffix": "cnn_comb",
        "result_folder": "cnn_result_files",
    },
    "xsum": {
        "finetune_data_folder": "xsum_comb",
        "eval_folder_suffix": "xsum_comb",
        "result_folder": "xsum_result_files",
    },
    "wikihow": {
        "finetune_data_folder": "wikihow_comb",
        "eval_folder_suffix": "wikihow_comb",
        "result_folder": "wikihow_result_files",
    },
}

###############################################################################
# API CLIENT (lazily -- a fully resumed/finished run must not need a key)
###############################################################################

client = None


def load_client_if_needed():
    global client

    if client is not None:
        return

    if OpenAI is None:
        raise SystemExit(
            "The 'openai' package is required to send requests.\n"
            "    pip install openai"
        )

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set.\n"
            "  1. create a key at https://platform.openai.com/api-keys\n"
            "  2. export OPENAI_API_KEY=\"sk-...\"\n"
            "A ChatGPT Plus subscription does NOT provide API access; the API is\n"
            "billed separately at https://platform.openai.com/settings/organization/billing"
        )

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    print(f"OpenAI client ready -- model={MODEL}, reasoning effort={REASONING_EFFORT}")


def is_reasoning_model(model: str) -> bool:
    return model.startswith(REASONING_MODEL_PREFIXES)


def request_parameters_for(model: str) -> dict:
    """
    The model-family-specific half of the request (see REASONING_MODEL_PREFIXES).
    """

    if is_reasoning_model(model):
        return {"reasoning": {"effort": REASONING_EFFORT}}

    # Non-reasoning model -> pin what this family does let us pin.
    return {"temperature": 0, "seed": SEED}


def price_of(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD for one call. Unknown models price as 0 rather than guessing."""

    rates = PRICING.get(model)
    if rates is None:
        return 0.0

    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000

###############################################################################
# RESULT PARSING
###############################################################################

def parse_prometheus_output(decoded_output: str):
    """
    Splits output into:
      - feedback (before [RESULT])
      - raw_result (A / B / TIE)

    "TIE_2" means the model never emitted a [RESULT] tag -- usually because the
    generation hit MAX_NEW_TOKENS before finishing. Those samples are counted as
    ties, so the rate is reported separately in the logs to make it obvious if
    MAX_NEW_TOKENS is set too low.
    """

    if "[RESULT]" not in decoded_output:
        return decoded_output.strip(), "TIE_2"

    split_output = decoded_output.split("[RESULT]")
    feedback = ""
    for output in split_output[:-1]:
        feedback += output

    result_part = split_output[-1]
    result_part = result_part.strip().upper()

    if result_part.startswith("A"):
        result = "A"
    elif result_part.startswith("B"):
        result = "B"
    elif result_part.startswith("TIE"):
        result = "TIE"
    else:
        result = "TIE"

    return feedback.strip(), result

###############################################################################
# JUDGE PROMPT (versus the REFERENCE SUMMARY)
###############################################################################

# Carried over verbatim from calc_LLM_as_a_judge_step5_Prometheus.py -- system
# message, task description, criteria and the closing note -- so that step 5 and
# step 7 answer exactly the same question and only the judge differs.
#
# ONE prompt for all three datasets, on purpose. The per-dataset split (news vs
# procedural) existed only because the source document was the ground truth
# there: a how-to text makes step order and exact quantities load-bearing in a
# way a news article does not. A reference summary carries none of that
# structure -- it is a couple of sentences of ground truth -- so the same four
# criteria apply to cnn, xsum and wikihow alike, and the three datasets stay
# directly comparable with one another.
#
# The other reason to prefer this prompt: judging faithfulness against the full
# document structurally rewards copying, because text lifted verbatim from the
# source cannot contradict it. Measured on wikihow/1M under the source-document
# prompt, the more extractive summary won 58% of decided pairs -- a poor test of
# a claim about abstractive quality. Consistency against a short reference does
# not have that failure mode.
#
# Consequence worth knowing: results under this prompt are NOT comparable with
# the earlier source-document step 7 runs, nor with step 6. They ARE the
# paid-judge counterpart of step 5.
#
# Those earlier runs are archived OUTSIDE this repo, in
#   /home/ege/Desktop/Pmi_Pegasus/llm judge GPT initial results/
# and are the reference point for positional bias on wikihow/1M, full 5577:
#   effort="none", source-doc prompt -> slot-A advantage +25.1 +/-2.6
#   effort="low",  source-doc prompt -> slot-A advantage +10.9 +/-2.6
# i.e. raising the effort bought a ~14-point reduction. Compare any new
# reference-summary number against +10.9, and only at full sample size -- see
# MAX_SAMPLES_PER_COMPARISON, a 20-sample run resolves this to about +/-44.

SYSTEM_MESSAGE = (
    "You are a fair and precise evaluation assistant. "
    "You compare two candidate summaries against a reference summary. "
    "Follow the evaluation criteria carefully and be impartial."
)


def build_judge_prompt(
    reference_summary: str,
    pmi_summary: str,
    rouge_summary: str,
    swap: bool,
):
    """
    Builds (system_message, user_message) for one sample.

    `swap` decides the position of each candidate, so that positional bias is
    averaged out. It is derived deterministically by the caller, which keeps a
    resumed run identical to an uninterrupted one.

    No truncation: reference summaries are a few hundred characters, so nothing
    here comes close to the context window (step 6 needed a 4000-token cut only
    because it fed whole documents to Prometheus).
    """

    if swap:
        response_A = pmi_summary
        response_B = rouge_summary
    else:
        response_A = rouge_summary
        response_B = pmi_summary

    instruction = f"""
TASK DESCRIPTION:
1. You are given a Reference Summary and two Candidate Summaries (A and B) of the same (unseen) source document.
2. Your task is to evaluate the quality of the two Candidate Summaries based on the Reference Summary using the specified Evaluation Criteria.
3. The source document is not shown to you. Treat the Reference Summary as the only ground truth, and do not speculate about content that might exist in the source document.
4. Write a brief feedback that assess the quality of the two candidate summaries strictly based on the given evaluation criteria, not evaluating in general.
5. After writing the feedback, indicate the better candidate summary, either "A" or "B" or "TIE".
6. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B" or "TIE")"
7. Please do not generate any other opening, closing, and explanations.

EVALUATION CRITERIA:
1. **Consistency with the Reference:** Does the summary avoid stating anything that contradicts the Reference Summary?
2. **Coverage:** How well does the summary capture the essential points mentioned in the Reference Summary?
3. **Conciseness:** Is the summary brief without sacrificing key details of the Reference Summary?
4. **Coherence:** Is the summary easy to read and logically organized?

Note: Extra details that are absent from the Reference Summary are not automatically errors, because the source document may contain them. Penalize such details only when they contradict the Reference Summary or clearly dilute its essential points.

REFERENCE SUMMARY:
{reference_summary}

CANDIDATE A:
{response_A}

CANDIDATE B:
{response_B}

FEEDBACK:
""".strip()

    return SYSTEM_MESSAGE, instruction


def swap_for_sample(dataset_key: str, checkpoint: str, sample_idx: int) -> bool:
    """
    Deterministic per-sample position swap.

    Step 5 drew the swap from the global RNG, so its positions cannot be
    reproduced here; seeding per sample instead makes a resumed run identical to
    an uninterrupted one, which is what matters when every call is paid for.
    Positional bias is averaged out either way.
    """

    return random.Random(f"{dataset_key}|{checkpoint}|{sample_idx}").random() < 0.5


def decode_winner(raw_result: str, swap: bool) -> str:
    """Maps the judge's A/B/TIE verdict back onto PMI / ROUGE."""

    if swap:
        return (
            "pmi" if raw_result == "A"
            else "rouge" if raw_result == "B"
            else "tie"
        )

    return (
        "rouge" if raw_result == "A"
        else "pmi" if raw_result == "B"
        else "tie"
    )

###############################################################################
# ONE API CALL
###############################################################################

def judge_one_with_retry(reference_summary, pmi_summary, rouge_summary, swap):
    """
    Judges a single sample, retrying on rate limits and transient errors.

    Returns (winner, feedback, raw_result, meta). A call that still fails after
    MAX_RETRIES raises: better to stop the run -- the partial file keeps every
    sample judged so far -- than to record a fabricated tie and quietly poison
    the aggregate.
    """

    system_message, user_message = build_judge_prompt(
        reference_summary, pmi_summary, rouge_summary, swap
    )

    request = {
        "model": MODEL,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "input": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    }
    request.update(request_parameters_for(MODEL))

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.responses.create(**request)
            break
        except (openai.RateLimitError, openai.APITimeoutError,
                openai.APIConnectionError, openai.InternalServerError) as error:
            last_error = error
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** (attempt + 1))
        except openai.BadRequestError as error:
            # A model that rejects one of the optional knobs (a family gaining
            # or losing support for seed/temperature/reasoning) should cost one
            # wasted call, not the whole run. Drop the named parameter once and
            # retry; anything else is a real error and propagates.
            dropped = next(
                (name for name in ("seed", "temperature", "reasoning")
                 if name in request and name in str(error)),
                None,
            )
            if dropped is None:
                raise
            print(f"[INFO] {MODEL} rejected '{dropped}' -- retrying without it")
            request.pop(dropped)
    else:  # pragma: no cover -- the loop either breaks or raises
        raise last_error

    completion = response.output_text
    feedback, raw_result = parse_prometheus_output(completion)

    usage = response.usage
    reasoning_tokens = 0
    details = getattr(usage, "output_tokens_details", None)
    if details is not None:
        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

    meta = {
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT if is_reasoning_model(MODEL) else None,
        "temperature": request.get("temperature"),
        "seed": request.get("seed"),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "response_id": response.id,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cost_usd": price_of(MODEL, usage.input_tokens, usage.output_tokens),
    }

    return decode_winner(raw_result, swap), feedback, raw_result, meta

###############################################################################
# SAMPLE SELECTION + COST ESTIMATE
###############################################################################

def select_sample_indices(dataset_key: str, total_rows: int, max_samples):
    """
    The documents to judge, as a sorted list of indices.

    Seeded by DATASET ONLY, deliberately: every checkpoint of a dataset is then
    judged on the same documents, so the 8 numbers of a dataset differ because
    the summaries differ, not because the sample did. A random subset rather
    than the first N, because the test sets are not shuffled.
    """

    if max_samples is None or max_samples >= total_rows:
        return list(range(total_rows))

    rng = random.Random(f"step7|{dataset_key}")
    return sorted(rng.sample(range(total_rows), max_samples))


def estimate_tokens(text: str) -> int:
    """
    Rough token count for the pre-flight estimate only.

    Deliberately not tiktoken: the estimate does not need to be exact, and this
    keeps the script from carrying a dependency that is otherwise unused. ~4
    characters per token is a good approximation for English prose; measured
    against a real run it comes out ~10% HIGH, which is the safe direction for a
    number whose job is to stop you from starting a run you did not intend to
    pay for. The real numbers are read back from the API's usage field and
    reported per comparison.
    """

    return len(text) // 4


def estimate_comparison(dataset_key: str, checkpoint: str, max_samples):
    """Projected tokens and cost for one comparison, without calling the API."""

    config = DATASETS[dataset_key]
    ds = load_from_disk(str(FINETUNE_DATA_DIR / config["finetune_data_folder"] / "test"))
    reference_summaries = ds["summary"]

    pmi_summaries, rouge_summaries = read_candidate_summaries(dataset_key, checkpoint)
    indices = select_sample_indices(dataset_key, len(reference_summaries), max_samples)

    input_tokens = 0
    for i in indices:
        system_message, user_message = build_judge_prompt(
            reference_summaries[i], pmi_summaries[i], rouge_summaries[i], True
        )
        input_tokens += estimate_tokens(system_message) + estimate_tokens(user_message)

    output_tokens = len(indices) * EXPECTED_OUTPUT_TOKENS

    return {
        "samples": len(indices),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": price_of(MODEL, input_tokens, output_tokens),
    }

###############################################################################
# RESUME HELPERS
###############################################################################

def read_partial_results(partial_path: Path):
    """
    Reads the already judged samples of an interrupted run.

    A run can be killed in the middle of writing a line, so a trailing broken
    line is simply dropped. Entries must be contiguous starting from index 0;
    anything after a gap is discarded.
    """

    if not partial_path.exists():
        return []

    entries = []
    with open(partial_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                break  # truncated last line
            if entry.get("position") != len(entries):
                break  # gap -> stop here and re-judge from this point on
            entries.append(entry)

    return entries


def rewrite_partial_file(partial_path: Path, entries):
    """Rewrites the partial file so it exactly matches the kept entries."""

    with open(partial_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_candidate_summaries(dataset_key: str, checkpoint: str):
    config = DATASETS[dataset_key]

    paths = {}
    for kind in ("PMI", "ROUGE"):
        paths[kind] = (
            GENERATED_PRED_DIR
            / f"eval_results_{kind}_pegasus_complete_{checkpoint}_pt_100k_ft_{config['eval_folder_suffix']}"
            / "generated_predictions.txt"
        )
        if not paths[kind].exists():
            raise FileNotFoundError(f"Missing required input: {paths[kind]}")

    summaries = {}
    for kind, path in paths.items():
        with open(path, "r", encoding="utf-8") as f:
            summaries[kind] = [line.strip() for line in f]

    return summaries["PMI"], summaries["ROUGE"]


def position_stratified_stats(entries):
    """
    PMI's win-rate with the two candidate slots weighted equally, instead of
    pooled.

    The A/B swap is an independent coin flip per sample, so the slots never come
    out exactly even, and this judge has a large positional preference (~25
    points on wikihow/1M). Pooling therefore over-weights whichever slot happened
    to get more draws: on wikihow/1M a 1.8% slot imbalance moved the pooled
    figure by 0.24 points, against a measured effect of only ~1.6 points.

    Averaging the two slot rates removes that, and -- more importantly across a
    24-comparison grid -- keeps the numbers comparable when the coin flip lands
    differently from one comparison to the next.

    Returns None when either slot has no decided pair (very small runs).
    """

    decided = [e for e in entries if e.get("llm_judge_winner") in ("pmi", "rouge")]

    slots = {}
    for slot in ("A", "B"):
        rows = [e for e in decided if e.get("pmi_position") == slot]
        if not rows:
            return None
        wins = sum(1 for e in rows if e["llm_judge_winner"] == "pmi")
        slots[slot] = (wins / len(rows), len(rows))

    (rate_a, n_a), (rate_b, n_b) = slots["A"], slots["B"]
    rate = (rate_a + rate_b) / 2
    se = math.sqrt((rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b) / 4)

    # ROUGE is reported alongside PMI for readability, but note it carries no
    # independent information: ties are excluded from `decided`, so on every
    # pair exactly one of the two systems wins and ROUGE's rate is 1 - PMI's.
    # Its slot-A advantage is identical to PMI's for the same reason -- when PMI
    # sits in A, ROUGE sits in B. Read the two rows as one measurement shown
    # from both sides, not as two agreeing measurements.
    return {
        "rate": rate, "half_ci": 1.96 * se,
        "rate_a": rate_a, "rate_b": rate_b, "n_a": n_a, "n_b": n_b,
        # ROUGE shown as A == the pairs where PMI sat in B, and vice versa.
        "rouge_rate": 1 - rate,
        "rouge_rate_a": 1 - rate_b, "rouge_rate_b": 1 - rate_a,
        "rouge_n_a": n_b, "rouge_n_b": n_a,
        "advantage": rate_a - rate_b, "decided": len(decided),
    }


def aggregate_text(dataset_key: str, checkpoint: str, entries) -> str:
    counts = Counter(entry["llm_judge_winner"] for entry in entries)
    total = len(entries)

    # Generations that never produced a [RESULT] tag. They fall into "tie", so
    # a high rate here means MAX_OUTPUT_TOKENS is cutting the judge off.
    unparsed = sum(1 for entry in entries if entry.get("raw_result") == "TIE_2")

    cost = sum(entry.get("cost_usd", 0.0) for entry in entries)
    input_tokens = sum(entry.get("input_tokens", 0) for entry in entries)
    output_tokens = sum(entry.get("output_tokens", 0) for entry in entries)
    models = sorted({entry.get("model", MODEL) for entry in entries})
    efforts = sorted({entry.get("reasoning_effort", REASONING_EFFORT) for entry in entries})

    lines = [
        f"DATASET   : {dataset_key}",
        f"CHECKPOINT: {checkpoint}",
        f"JUDGE     : {','.join(models)} via OpenAI API (compared against the REFERENCE SUMMARIES)",
        f"PROMPT    : step 5 reference-summary prompt (same for every dataset)",
        f"SETTINGS  : reasoning_effort={','.join(str(e) for e in efforts)}, "
        f"max_output_tokens={MAX_OUTPUT_TOKENS}, "
        f"temperature={sorted({str(e.get('temperature')) for e in entries})[0]} "
        f"(reasoning models pin it at 1 and reject the parameter)",
        f"SAMPLES   : {total} (subsets are a deterministic random sample, "
        f"seeded per dataset so every checkpoint sees the same documents)",
        f"TOKENS    : input={input_tokens}, output={output_tokens}",
        f"COST      : ${cost:.2f}",
        "----------------------",
        f"PMI wins   : {counts['pmi']}  ({counts['pmi'] / total * 100:.4f}%)",
        f"ROUGE wins : {counts['rouge']} ({counts['rouge'] / total * 100:.4f}%)",
        f"TIES       : {counts['tie']} ({counts['tie'] / total * 100:.4f}%)",
        "----------------------",
        f"  of which no [RESULT] tag : {unparsed} ({unparsed / total * 100:.4f}%)",
    ]

    stats = position_stratified_stats(entries)
    if stats is not None:
        lines += [
            "----------------------",
            f"POSITION-STRATIFIED (slots weighted equally; {stats['decided']} decided pairs, ties excluded)",
            f"{'':13s}{'win% shown as A':>22s}{'win% shown as B':>22s}{'stratified win%':>22s}",
            f"  {'PMI':11s}"
            f"{stats['rate_a'] * 100:>14.4f} (n={stats['n_a']:d})"
            f"{stats['rate_b'] * 100:>14.4f} (n={stats['n_b']:d})"
            f"{stats['rate'] * 100:>15.4f} +/-{stats['half_ci'] * 100:.4f}",
            f"  {'ROUGE':11s}"
            f"{stats['rouge_rate_a'] * 100:>14.4f} (n={stats['rouge_n_a']:d})"
            f"{stats['rouge_rate_b'] * 100:>14.4f} (n={stats['rouge_n_b']:d})"
            f"{stats['rouge_rate'] * 100:>15.4f} +/-{stats['half_ci'] * 100:.4f}",
            f"  slot-A advantage : {stats['advantage'] * 100:+.4f} points for whichever system sits in A",
            f"    (large values mean position, not content, drove the individual verdicts)",
            f"  NOTE: ties are excluded here, so ROUGE is exactly 100 - PMI -- the same",
            f"        measurement shown from both sides, not independent confirmation.",
            f"  -> prefer these over the pooled figures above when comparing checkpoints",
        ]

    return "\n".join(lines)


def write_dataset_summary(dataset_key: str):
    """
    (Re)builds the per-dataset summary log from whatever comparison outputs are
    actually on disk, so a grid split across several runs stays complete.
    """

    result_dir = SCRIPT_DIR / DATASETS[dataset_key]["result_folder"]

    lines = [
        f"LLM-as-a-judge (OpenAI API) versus REFERENCE SUMMARIES -- step 7",
        f"DATASET: {dataset_key}",
        f"PROMPT : step 5 reference-summary prompt (same for every dataset)",
        "=" * 70,
    ]

    grand_total_cost = 0.0

    for checkpoint in CHECKPOINTS:
        path = result_dir / f"{dataset_key}_{checkpoint}_llm_judge_gpt_vs_reference_summaries__step7.json"

        if not path.exists():
            lines.append(f"{checkpoint}: (not run yet)")
            continue

        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        if not entries:
            lines.append(f"{checkpoint}: (empty output file)")
            continue

        counts = Counter(entry["llm_judge_winner"] for entry in entries)
        unparsed = sum(1 for entry in entries if entry.get("raw_result") == "TIE_2")
        cost = sum(entry.get("cost_usd", 0.0) for entry in entries)
        grand_total_cost += cost
        total = len(entries)

        lines.append(
            f"{checkpoint}: samples={total} | "
            f"pmi={counts['pmi']} ({counts['pmi'] / total * 100:.4f}%) | "
            f"rouge={counts['rouge']} ({counts['rouge'] / total * 100:.4f}%) | "
            f"tie={counts['tie']} ({counts['tie'] / total * 100:.4f}%) | "
            f"no_result_tag={unparsed} ({unparsed / total * 100:.4f}%) | "
            f"${cost:.2f}"
        )

        stats = position_stratified_stats(entries)
        if stats is not None:
            lines.append(
                f"    position-stratified: pmi={stats['rate'] * 100:.4f}% "
                f"+/-{stats['half_ci'] * 100:.4f} | "
                f"rouge={stats['rouge_rate'] * 100:.4f}% | "
                f"slot-A advantage={stats['advantage'] * 100:+.2f} pts"
            )

    lines.append("=" * 70)
    lines.append(f"TOTAL SPENT ON THIS DATASET: ${grand_total_cost:.2f}")

    summary_path = (
        result_dir
        / f"{dataset_key}_ALL_checkpoints_llm_judge_gpt_vs_reference_summaries__step7_summary.log"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

###############################################################################
# COMMAND LINE
###############################################################################

def select_datasets(raw: str):
    if raw.strip().lower() == "all":
        return list(DATASETS)

    selected = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in DATASETS:
            raise SystemExit(
                f"Unknown dataset '{token}'. Valid choices: {', '.join(DATASETS)}, all"
            )
        if token not in selected:
            selected.append(token)

    if not selected:
        raise SystemExit("--datasets did not select anything.")

    return selected


def select_checkpoints(raw: str):
    if raw.strip().lower() == "all":
        return list(CHECKPOINTS)

    selected = []
    for token in raw.split(","):
        token = token.strip().upper()
        if not token:
            continue
        if not token.endswith("M"):
            token += "M"  # allow "1,2" as well as "1M,2M"
        if token not in CHECKPOINTS:
            raise SystemExit(
                f"Unknown checkpoint '{token}'. Valid choices: {', '.join(CHECKPOINTS)}, all"
            )
        if token not in selected:
            selected.append(token)

    if not selected:
        raise SystemExit("--checkpoints did not select anything.")

    return selected


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Step 7 -- paid-API LLM-as-a-judge of PMI- vs ROUGE-pegasus summaries "
            "against the reference summaries. Same prompt and same outputs as "
            "step 5, different judge. RUN --estimate FIRST: the full grid is "
            "227208 paid calls."
        )
    )
    parser.add_argument("--datasets", default="all",
                        help=f"comma separated subset of: {', '.join(DATASETS)} (default: all)")
    parser.add_argument("--checkpoints", default="all",
                        help=f"comma separated subset of: {', '.join(CHECKPOINTS)} (default: all)")
    parser.add_argument("--model", default=MODEL,
                        help=f"exact model id, never an alias (default: {MODEL}). "
                             f"Priced: {', '.join(PRICING)}")
    parser.add_argument("--effort", default=REASONING_EFFORT,
                        choices=["none", "low", "medium", "high", "xhigh", "max"],
                        help=f"reasoning effort; billed at the output rate, and "
                             f"ignored for non-reasoning models such as gpt-4o-mini "
                             f"(default: {REASONING_EFFORT})")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_PER_COMPARISON,
                        help="samples judged per comparison; 0 or omitted = the whole "
                             "test set, which is what steps 4/5/6 used and the only "
                             "setting directly comparable with them (default: whole test set)")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
                        help=f"parallel in-flight requests; cannot affect verdicts "
                             f"(default: {CONCURRENCY})")
    parser.add_argument("--estimate", action="store_true",
                        help="print the projected tokens and cost, then exit without calling the API")

    args = parser.parse_args()

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1.")
    if args.max_samples is not None and args.max_samples < 0:
        raise SystemExit("--max-samples must be 0 (whole test set) or a positive number.")

    return args

###############################################################################
# SINGLE COMPARISON (one dataset + one checkpoint)
###############################################################################

def run_single_comparison(dataset_key: str, checkpoint: str, max_samples, concurrency: int):
    config = DATASETS[dataset_key]

    result_dir = SCRIPT_DIR / config["result_folder"]
    result_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{dataset_key}_{checkpoint}_llm_judge_gpt_vs_reference_summaries__step7"
    output_path = result_dir / f"{base_name}.json"
    partial_path = result_dir / f"{base_name}.partial.jsonl"
    log_path = result_dir / f"{base_name}.log"

    test_set_path = FINETUNE_DATA_DIR / config["finetune_data_folder"] / "test"
    if not test_set_path.exists():
        raise FileNotFoundError(f"Missing required input: {test_set_path}")

    ds = load_from_disk(str(test_set_path))
    reference_summaries = ds["summary"]
    doc_ids = ds["id"] if "id" in ds.column_names else list(range(len(ds)))

    pmi_summaries, rouge_summaries = read_candidate_summaries(dataset_key, checkpoint)

    if not (len(reference_summaries) == len(pmi_summaries) == len(rouge_summaries)):
        raise ValueError(
            f"Size mismatch for {dataset_key} {checkpoint}: "
            f"references={len(reference_summaries)}, pmi={len(pmi_summaries)}, "
            f"rouge={len(rouge_summaries)}"
        )

    indices = select_sample_indices(dataset_key, len(reference_summaries), max_samples)
    total_samples = len(indices)

    # ---- already finished? -------------------------------------------------
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            finished_entries = json.load(f)
        if len(finished_entries) == total_samples:
            print(f"[SKIP] {dataset_key} {checkpoint}: already complete "
                  f"({total_samples} samples, $0.00 spent) -> {output_path.name}")
            return finished_entries
        print(f"[REDO] {dataset_key} {checkpoint}: existing output has "
              f"{len(finished_entries)} / {total_samples} samples, continuing...")

    # ---- resume from the partial file --------------------------------------
    # Every entry here was PAID FOR, so the cache is never discarded lightly:
    # unlike step 5/6 there is no batch grouping to realign, because each call
    # is independent.
    entries = read_partial_results(partial_path)
    entries = entries[:total_samples]
    rewrite_partial_file(partial_path, entries)

    start_index = len(entries)
    if start_index > 0:
        print(f"[RESUME] {dataset_key} {checkpoint}: continuing at sample "
              f"{start_index} / {total_samples} "
              f"(${sum(e.get('cost_usd', 0.0) for e in entries):.2f} already spent)")

    if start_index < total_samples:
        load_client_if_needed()

    partial_file = open(partial_path, "a", encoding="utf-8")
    try:
        progress = tqdm(
            total=total_samples,
            initial=start_index,
            desc=f"{dataset_key} {checkpoint} (conc={concurrency})",
        )

        chunk_size = concurrency * 4
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for chunk_start in range(start_index, total_samples, chunk_size):
                positions = list(
                    range(chunk_start, min(chunk_start + chunk_size, total_samples))
                )

                futures = {}
                for position in positions:
                    i = indices[position]
                    swap = swap_for_sample(dataset_key, checkpoint, i)
                    futures[position] = pool.submit(
                        judge_one_with_retry,
                        reference_summaries[i], pmi_summaries[i], rouge_summaries[i],
                        swap,
                    )

                # Collected in submission order, not completion order, so the
                # partial file stays contiguous and a resume is unambiguous.
                for position in positions:
                    i = indices[position]
                    swap = swap_for_sample(dataset_key, checkpoint, i)
                    winner, feedback, raw_result, meta = futures[position].result()

                    entry = {
                        "position": position,      # index within the judged subset
                        "index": i,                # row in the test set
                        "id": doc_ids[i],
                        "reference_summary": reference_summaries[i],
                        "pmi_summary": pmi_summaries[i],
                        "rouge_summary": rouge_summaries[i],
                        "pmi_position": "A" if swap else "B",
                        "llm_judge_winner": winner,
                        "raw_result": raw_result,
                        "llm_judge_feedback": feedback,
                    }
                    entry.update(meta)

                    entries.append(entry)
                    partial_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

                # Flushed once per chunk: an interrupted run loses at most the
                # chunk in flight, and every flushed sample is money kept.
                partial_file.flush()
                os.fsync(partial_file.fileno())

                spent = sum(e.get("cost_usd", 0.0) for e in entries)
                progress.set_postfix_str(f"${spent:.2f}")
                progress.update(len(positions))

        progress.close()
    finally:
        partial_file.close()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)

    aggregate = aggregate_text(dataset_key, checkpoint, entries)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(aggregate + "\n")

    print("\n" + aggregate + "\n")

    partial_path.unlink(missing_ok=True)

    return entries

###############################################################################
# MAIN ENTRY POINT
###############################################################################

if __name__ == "__main__":

    args = parse_args()
    selected_datasets = select_datasets(args.datasets)
    selected_checkpoints = select_checkpoints(args.checkpoints)

    MODEL = args.model
    REASONING_EFFORT = args.effort
    max_samples = None if args.max_samples in (0, None) else args.max_samples

    if MODEL not in PRICING:
        print(f"[WARN] no price on file for '{MODEL}' -- cost will be reported as $0.00.\n"
              f"       Add it to PRICING from https://developers.openai.com/api/docs/pricing")

    print(f"Datasets   : {', '.join(selected_datasets)}")
    print(f"Checkpoints: {', '.join(selected_checkpoints)}")
    print(f"Model      : {MODEL} (effort={REASONING_EFFORT})")
    print(f"Samples    : {'whole test set' if max_samples is None else max_samples} per comparison")
    print(f"=> {len(selected_datasets) * len(selected_checkpoints)} comparison(s)")

    # ---- pre-flight cost estimate -----------------------------------------
    if args.estimate:
        print("\nESTIMATE ONLY -- no API calls will be made")
        print("=" * 70)
        grand = {"samples": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        for dataset_key in selected_datasets:
            for checkpoint in selected_checkpoints:
                est = estimate_comparison(dataset_key, checkpoint, max_samples)
                for key in grand:
                    grand[key] += est[key]
                print(f"{dataset_key:8s} {checkpoint:3s} | samples={est['samples']:6d} | "
                      f"in={est['input_tokens'] / 1e6:7.2f}M | "
                      f"out={est['output_tokens'] / 1e6:6.2f}M | ${est['cost_usd']:8.2f}")
        print("=" * 70)
        print(f"TOTAL: {grand['samples']} calls, "
              f"{grand['input_tokens'] / 1e6:.1f}M input + {grand['output_tokens'] / 1e6:.1f}M output "
              f"tokens, ${grand['cost_usd']:.2f}")
        if REASONING_EFFORT != EXPECTED_OUTPUT_TOKENS_MEASURED_AT_EFFORT:
            print(f"\n[WARN] this estimate ASSUMES {EXPECTED_OUTPUT_TOKENS} output tokens/verdict, "
                  f"measured at effort='{EXPECTED_OUTPUT_TOKENS_MEASURED_AT_EFFORT}',")
            print(f"       but you are running effort='{REASONING_EFFORT}'. Reasoning tokens bill at the")
            print(f"       OUTPUT rate, so the real cost may be SEVERAL TIMES this figure.")
            print(f"       Measure it first:  --datasets wikihow --checkpoints 1M --max-samples 20")
        print(f"\nOutput assumed at {EXPECTED_OUTPUT_TOKENS} tokens/verdict (measured on a real run);")
        print("input is estimated ~10% high. Real usage is reported per comparison.")
        print("The Batch API halves both rates if a ~24h turnaround is acceptable.")
        raise SystemExit(0)

    overall_summary = {}

    for dataset_key in selected_datasets:
        for checkpoint in selected_checkpoints:
            print(f"\n\n{'*' * 70}")
            print(f"***  {dataset_key.upper()}  --  {checkpoint} pretraining steps")
            print(f"{'*' * 70}\n")

            entries = run_single_comparison(
                dataset_key, checkpoint, max_samples, args.concurrency
            )

            overall_summary[(dataset_key, checkpoint)] = Counter(
                entry["llm_judge_winner"] for entry in entries
            )

            write_dataset_summary(dataset_key)

    print("\n\nOVERALL RESULTS (PMI / ROUGE / TIE)")
    print("=" * 70)
    for (dataset_key, checkpoint), counts in overall_summary.items():
        total = sum(counts.values())
        print(
            f"{dataset_key:8s} {checkpoint:3s} | "
            f"pmi={counts['pmi']:6d} ({counts['pmi'] / total * 100:7.4f}%) | "
            f"rouge={counts['rouge']:6d} ({counts['rouge'] / total * 100:7.4f}%) | "
            f"tie={counts['tie']:6d} ({counts['tie'] / total * 100:7.4f}%)"
        )
    print("=" * 70)

"""
CORRELATION CHECK -- paid GPT judge vs. the 105-item HUMAN EVALUATION set.

The human study (see "HUMAN Eval/prepare_human_eval_set.py") shows three
annotators 105 items -- 35 each from cnn, wikihow and xsum, all at the 8M
checkpoint -- and asks two separate questions per item:

    BEST ACCORDING TO FAITHFULNESS      (A / B / tie)
    BEST ACCORDING TO INFORMATIVENESS   (A / B / tie)

This script puts the SAME 105 items, in the SAME candidate arrangement, to
gpt-5.6-luna, one question at a time: 105 faithfulness calls + 105
informativeness calls = 210 paid calls per full run. It then measures how well
the paid judge agrees with the humans.

WHY THE ARRANGEMENT MATTERS
---------------------------
Both the items and their A/B order are taken from the files the annotators
actually received:

    human_eval_set.txt   the exact source document, reference summary and the
                         two candidate texts, read back verbatim
    answer_key.txt       which system (PMI / ROUGE) sat in slot A and slot B

Nothing is re-sampled and nothing is re-shuffled here. If the judge and the
humans saw different arrangements, any positional bias in either would leak
straight into the correlation, so the arrangement is treated as fixed input,
not as something this script gets to decide. build_items() cross-checks the two
files against each other and refuses to run if they disagree.

THE TWO PROMPTS
---------------
One prompt per dimension, each carrying a SINGLE criterion, worded to match the
definition the annotators were given. Everything else -- the system message,
the numbered task description, the "[RESULT]" contract and its parser -- is
step 7's, so the judge answers the same kind of question in the same format.

Unlike step 5/7 (reference summary only) the source document IS shown, because
that is what the humans were asked to judge against; the reference summary is
included too, and framed as a guide, exactly as in their instructions.

DETERMINISM
-----------
Same as step 7, and for the same reason: the GPT-5 series fixes temperature at
1 and takes no seed, so per-call verdicts are stochastic and no setting changes
that. What is pinned is the exact model id, the reasoning effort, the prompts
and the candidate positions -- and the results themselves, since a judged
(item, dimension) pair is never asked twice: the ".partial.jsonl" is the cache.

COST
----
210 calls over ~105 source documents (~55k words, read twice). At effort=low on
gpt-5.6-luna that is roughly 170k input + 45k output tokens, about $0.09 --
three orders of magnitude below the step 7 grid, because this is 210 calls
rather than 227208. --estimate still exists and still runs first.

SETUP
-----
  1. pip install openai
  2. export OPENAI_API_KEY="sk-..."      (never hard-code it here)
  3. copy human_eval_set.txt (the FINAL_VERSION copy) and answer_key.txt next to
     this script -- see EVAL_SET_PATH / ANSWER_KEY_PATH
  4. python GPT_API_corralation_check_with_human_eval.py --estimate
  5. python GPT_API_corralation_check_with_human_eval.py
  6. once the annotators are done:
     python GPT_API_corralation_check_with_human_eval.py --report-only \
            --human-answers votes.csv

Step 6 collects the human votes. See load_human_answers() for the CSV format;
the correlation section is skipped entirely until that file is supplied, so
steps 4-5 are useful on their own, and adding the votes later costs nothing
because judged pairs are cached.

THE SURFACE NOTE
----------------
The prompt carries one extra sentence telling the judge to ignore the truncated
opening fragments these models emit, mirroring rule 1 of the annotators' notes.
Whether it belongs is a genuine methodological question -- see the long comment
at SURFACE_NOTE_ENABLED -- so it is a switch, not a decision baked into the file:

    python GPT_API_corralation_check_with_human_eval.py --out-dir with_note
    python GPT_API_corralation_check_with_human_eval.py --no-surface-note \
           --out-dir without_note

At ~$0.11 a run the pair costs about $0.22 and settles the question with data.
Every record stores which way it was run, and the log says so at the top.
"""

import argparse
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

# Imported softly so that --estimate and --report-only (neither of which sends
# anything) still run on a machine without the SDK. The hard failure happens in
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

# Pinned to the exact model, deliberately NOT the "gpt-5.6" alias -- that alias
# routes to Sol at 25x the price and OpenAI can repoint it without notice. Same
# choice, and the same reasoning, as step 7.
MODEL = "gpt-5.6-luna"

# USD per 1M tokens, standard (non-batch) tier. Kept in sync with step 7.
PRICING = {
    "gpt-5-nano":    {"input": 0.05, "output": 0.40},
    "gpt-4.1-nano":  {"input": 0.10, "output": 0.40},
    "gpt-4o-mini":   {"input": 0.15, "output": 0.60},
    "gpt-5-mini":    {"input": 0.25, "output": 2.00},
    "gpt-4.1-mini":  {"input": 0.40, "output": 1.60},
    "gpt-5.4-nano":  {"input": 0.20, "output": 1.25},
    "gpt-5.6-luna":  {"input": 0.20, "output": 1.20, "cached_input": 0.02},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00, "cached_input": 0.20},
    "gpt-5.6-sol":   {"input": 5.00, "output": 30.00, "cached_input": 0.50},
    "gpt-5.5":       {"input": 5.00, "output": 30.00, "cached_input": 0.50},
}

# GPT-5/o-series are reasoning models: they take reasoning.effort and reject
# temperature/seed. GPT-4 series is the other way round. See step 7.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
SEED = 0

# "low" rather than "none", matching the step 7 setting that measurably halved
# this judge's slot-A preference (+25.1 -> +10.9 points on wikihow/1M). A
# correlation study is exactly where that bias would do the most damage, and
# 210 calls is cheap enough that the extra reasoning tokens do not matter.
REASONING_EFFORT = "low"

# Feedback plus the [RESULT] tag; also caps reasoning tokens. Step 7's value,
# where nothing came within range of it at this effort.
MAX_OUTPUT_TOKENS = 768

# Parallel in-flight requests. Purely a throughput knob -- each call is judged
# independently, so concurrency cannot change a verdict.
CONCURRENCY = 8

# Retry policy for 429s and transient 5xx. Sleeps 2, 4, 8, 16, 32 seconds.
MAX_RETRIES = 5

# Output tokens per verdict, for the --estimate projection only. Step 7's
# measured figure at effort="low" (209 = ~117 reasoning + ~92 visible). The
# single-criterion prompts here should land at or below that, so the estimate
# errs high, which is the safe direction. Re-measure from a completed .log.
EXPECTED_OUTPUT_TOKENS = 209

SCRIPT_DIR = Path(__file__).resolve().parent

# Inputs and outputs both live beside this file. Copy the two packet files here
# before the first run:
#
#   human_eval_set.txt   the packet the annotators actually received -- use the
#                        FINAL_VERSION copy, not the generator's raw output. The
#                        two differ only in the instruction header (deadline,
#                        reworded criteria, the NOTES ON JUDGING block); all 105
#                        items below the header are byte identical, so the
#                        candidate arrangement is the same either way. Taking the
#                        real packet keeps that guarantee honest if it ever stops
#                        being true.
#
#   answer_key.txt       which system sat in slot A / slot B. Never shown to the
#                        annotators, and never written to by this script.
#
# Override either with --eval-set / --answer-key without moving anything.
EVAL_SET_PATH = SCRIPT_DIR / "human_eval_set.txt"
ANSWER_KEY_PATH = SCRIPT_DIR / "answer_key.txt"

DEFAULT_OUT_DIR = SCRIPT_DIR
BASE_NAME = "gpt_judge_vs_human_eval"

# The two questions, in the order they are asked of the annotators.
#
# The criterion text tracks the FINAL_VERSION packet, including the sentence
# each definition ends on -- faithfulness leans on the source document, while
# informativeness weighs the reference summary and the source document equally.
# That asymmetry is the whole reason the reference is not framed once, globally,
# in the task description: the two dimensions genuinely give it different
# weight, and a judge told otherwise would be answering a different question
# from the humans, which shows up as lost correlation rather than as an error.
#
# KEEP THESE IN SYNC with the packet. If the annotators' wording changes, change
# it here too.
DIMENSIONS = {
    "faithfulness": (
        "**Faithfulness:** Which candidate summary is better supported by the "
        "Source Document? Penalize anything the Source Document does not say: "
        "invented facts, wrong names, dates or numbers, and garbled or "
        "self-contradictory claims. Focus mainly on the consistency of the "
        "candidates with the Source Document."
    ),
    "informativeness": (
        "**Informativeness:** Which candidate summary covers more of the "
        "important content of the Source Document? Penalize essential points "
        "that are left out, and padding (unnecessary additions) that carries "
        "no information. Weigh the Reference Summary and the Source Document "
        "equally when judging what counts as important."
    ),
}

###############################################################################
# API CLIENT (lazily -- --estimate and --report-only must not need a key)
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
            "billed separately."
        )

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    print(f"OpenAI client ready -- model={MODEL}, reasoning effort={REASONING_EFFORT}")


def is_reasoning_model(model):
    return model.startswith(REASONING_MODEL_PREFIXES)


def request_parameters_for(model):
    if is_reasoning_model(model):
        return {"reasoning": {"effort": REASONING_EFFORT}}
    return {"temperature": 0, "seed": SEED}


def price_of(model, input_tokens, output_tokens):
    """USD for one call. Unknown models price as 0 rather than guessing."""

    rates = PRICING.get(model)
    if rates is None:
        return 0.0
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000

###############################################################################
# READING THE HUMAN-EVAL PACKET
###############################################################################

# "    1      1  cnn         7162  ROUGE         PMI"
KEY_ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\w+)\s+(\d+)\s+(PMI|ROUGE)\s+(PMI|ROUGE)\s*$"
)

# The item separator in the packet: "ITEM 7" under a rule of 80 dashes.
ITEM_SPLIT = re.compile(r"^ITEM (\d+)\n-{80}\n", re.M)


def parse_answer_key(path):
    """-> {item number: {group, dataset, row, system_a, system_b}}"""

    if not path.exists():
        raise SystemExit(f"answer key not found: {path}")

    key = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = KEY_ROW.match(line)
        if not match:
            continue
        item = int(match.group(1))
        key[item] = {
            "group": int(match.group(2)),
            "dataset": match.group(3),
            "row": int(match.group(4)),
            "system_a": match.group(5),
            "system_b": match.group(6),
        }

    if not key:
        raise SystemExit(f"no answer-key rows parsed from {path}")
    return key


def parse_eval_set(path):
    """
    -> {item number: {document, reference, candidate_a, candidate_b}}

    Read back verbatim from the packet rather than regenerated from the test
    sets, so the judge is shown byte-for-byte what the annotators were shown.
    """

    if not path.exists():
        raise SystemExit(f"human eval set not found: {path}")

    text = path.read_text(encoding="utf-8")
    chunks = ITEM_SPLIT.split(text)

    items = {}
    for i in range(1, len(chunks), 2):
        number = int(chunks[i])
        body = chunks[i + 1]

        def field(label, following):
            match = re.search(
                r"^%s\n(.*?)\n\n(?:%s)" % (re.escape(label), following),
                body, re.S | re.M,
            )
            return match.group(1) if match else None

        parsed = {
            "document": field("SOURCE DOCUMENT:", "REFERENCE SUMMARY:"),
            "reference": field("REFERENCE SUMMARY:", "CANDIDATE A:"),
            "candidate_a": field("CANDIDATE A:", "CANDIDATE B:"),
            "candidate_b": field("CANDIDATE B:", r"BEST ACCORDING TO|YOUR CHOICE"),
        }
        if any(value is None for value in parsed.values()):
            raise SystemExit(f"item {number} in {path.name} could not be parsed")
        items[number] = parsed

    if not items:
        raise SystemExit(f"no items parsed from {path}")
    return items


def build_items(eval_set_path, answer_key_path):
    """
    Merges the packet and its answer key into one list, and validates them
    against each other.

    The two files come from the same generator, so a mismatch means one of them
    has been edited or regenerated on its own -- which would silently break the
    whole premise of this script. Better to stop here than to correlate against
    the wrong arrangement.
    """

    key = parse_answer_key(answer_key_path)
    texts = parse_eval_set(eval_set_path)

    missing = set(key) ^ set(texts)
    if missing:
        raise SystemExit(
            f"{answer_key_path.name} and {eval_set_path.name} cover different "
            f"items: {sorted(missing)[:10]}"
        )

    items = []
    for number in sorted(key):
        entry = dict(key[number])
        entry.update(texts[number])
        entry["item"] = number
        if entry["system_a"] == entry["system_b"]:
            raise SystemExit(f"item {number}: same system in both slots")
        items.append(entry)

    return items

###############################################################################
# JUDGE PROMPTS -- one criterion each
###############################################################################

SYSTEM_MESSAGE = (
    "You are a fair and precise evaluation assistant. "
    "You compare two candidate summaries of a source document on a single "
    "criterion. Follow the evaluation criterion carefully and be impartial."
)

# ---------------------------------------------------------------------------
# THE SURFACE NOTE -- one sentence, and a switch to remove it
# ---------------------------------------------------------------------------
# Rule 1 of the packet's "NOTES ON JUDGING", and nothing else. The annotators are
# told to ignore the truncated opening fragments these models emit; a judge that
# is not told the same thing is answering a slightly different question.
#
# It is kept because the artifacts are NOT evenly spread. Measured over the 105
# items:
#
#     both candidates affected      42  (40%)   symmetric, cancels out
#     exactly one affected          57  (54%)   the items where it can bias
#     neither                        6   (6%)
#
#   and of those 57 asymmetric items the affected candidate is PMI in 46 and
#   ROUGE in only 11 -- roughly 4:1 against PMI.
#
# Since the faithfulness criterion lists "garbled or self-contradictory claims"
# as something to penalise, a judge may read a truncated opening as exactly that,
# and would then be marking PMI down on a detokenisation artifact in 46 items
# where the humans were instructed not to.
#
# The counter-argument is just as real: steps 5/6/7 carry no such note, so a
# judge that has one is not the judge those steps deploy. That is what
# --no-surface-note is for. The run is ~$0.11, so settle it by measurement
# rather than by argument: run both into separate --out-dir folders and compare.
# Every record stores which way it was run, so the two cannot be confused.
#
# What is deliberately NOT in here, having been cut as unjustified:
#   * length guidance -- defensible, but not grounded in anything measured here;
#   * the line-break note -- the model handles that layout without help;
#   * "answer TIE rather than forcing a preference" -- the worst of the three.
#     Tie rate drives both agreement and kappa, so an instruction that nudges it
#     toward the human tie rate is tuning the judge toward the statistic being
#     reported.
#
# XSum is deliberately not named anywhere in the prompt either: the humans read
# one sheet covering all three groups, but naming a dataset inside a per-item
# prompt would hand the judge a hint about which group it is looking at.
SURFACE_NOTE_ENABLED = True

SURFACE_NOTE = (
    "Note: Both candidates come from small models and may begin with stray "
    "punctuation or contain a truncated or malformed word. Ignore these surface "
    "artifacts and judge what the summary states."
)


def build_judge_prompt(dimension, document, reference, candidate_a, candidate_b,
                       surface_note=None):
    """
    Builds (system_message, user_message) for one item on one dimension.

    No swap argument, and that is the point: the candidates are placed exactly
    as the packet placed them, so `candidate_a` is whatever the annotators saw
    as Candidate A. decode_winner() maps the verdict back onto PMI/ROUGE using
    the answer key.

    `surface_note` defaults to SURFACE_NOTE_ENABLED; pass False (or use
    --no-surface-note) to drop it and leave the prompt at step 7's shape. The
    note is appended to the criterion block rather than sitting in its own slot,
    so removing it leaves no stray blank line behind.

    No truncation either. The longest document in the packet is ~1900 words,
    well inside the context window -- step 6 needed a 4000-token cut only
    because it fed whole documents to a local 7B model.
    """

    if surface_note is None:
        surface_note = SURFACE_NOTE_ENABLED

    criterion = DIMENSIONS[dimension]
    if surface_note:
        criterion += "\n\n" + SURFACE_NOTE

    instruction = f"""
TASK DESCRIPTION:
1. You are given a Source Document, a Reference Summary, and two Candidate Summaries (A and B) of that Source Document.
2. Your task is to decide which Candidate Summary is better on the single Evaluation Criterion below, and on nothing else.
3. The Reference Summary is a human-written summary of the same Source Document. The Evaluation Criterion below states how much weight to give it.
4. Write a brief feedback that assesses the two candidate summaries strictly on the given criterion, not evaluating in general.
5. After writing the feedback, indicate the better candidate summary, either "A" or "B" or "TIE".
6. The output format should look as follows: "Feedback: (write a feedback for the criterion) [RESULT] (Either "A" or "B" or "TIE")"
7. Please do not generate any other opening, closing, and explanations.

EVALUATION CRITERION:
{criterion}

SOURCE DOCUMENT:
{document}

REFERENCE SUMMARY:
{reference}

CANDIDATE A:
{candidate_a}

CANDIDATE B:
{candidate_b}

FEEDBACK:
""".strip()

    return SYSTEM_MESSAGE, instruction


def parse_judge_output(decoded_output):
    """
    Splits the completion into (feedback, raw_result).

    "TIE_2" means no [RESULT] tag was emitted at all. Those count as ties but
    are reported separately, so it is obvious if MAX_OUTPUT_TOKENS is cutting
    the judge off. Carried over from step 5/7 unchanged.
    """

    if "[RESULT]" not in decoded_output:
        return decoded_output.strip(), "TIE_2"

    parts = decoded_output.split("[RESULT]")
    feedback = "".join(parts[:-1])
    tail = parts[-1].strip().upper()

    if tail.startswith("A"):
        result = "A"
    elif tail.startswith("B"):
        result = "B"
    else:
        result = "TIE"

    return feedback.strip(), result


def decode_winner(raw_result, system_a, system_b):
    """Maps the judge's A/B/TIE verdict onto pmi / rouge / tie."""

    if raw_result == "A":
        return system_a.lower()
    if raw_result == "B":
        return system_b.lower()
    return "tie"

###############################################################################
# ONE API CALL
###############################################################################

def judge_one_with_retry(item, dimension):
    """
    Judges one (item, dimension) pair, retrying on rate limits and transient
    errors. A call that still fails after MAX_RETRIES raises: the partial file
    keeps everything judged so far, which is better than recording a fabricated
    tie and quietly poisoning the aggregate.
    """

    system_message, user_message = build_judge_prompt(
        dimension, item["document"], item["reference"],
        item["candidate_a"], item["candidate_b"], SURFACE_NOTE_ENABLED,
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
            # A model that rejects one of the optional knobs should cost one
            # wasted call, not the whole run.
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

    feedback, raw_result = parse_judge_output(response.output_text)

    usage = response.usage
    details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0 if details else 0

    return {
        "item": item["item"],
        "group": item["group"],
        "dataset": item["dataset"],
        "row": item["row"],
        "dimension": dimension,
        "system_a": item["system_a"],
        "system_b": item["system_b"],
        "raw_result": raw_result,
        "gpt_winner": decode_winner(raw_result, item["system_a"], item["system_b"]),
        "feedback": feedback,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT if is_reasoning_model(MODEL) else None,
        # Recorded per call so two runs -- one with the note, one without --
        # can never be mistaken for each other after the fact.
        "surface_note": SURFACE_NOTE_ENABLED,
        "response_id": response.id,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cost_usd": price_of(MODEL, usage.input_tokens, usage.output_tokens),
    }

###############################################################################
# COST ESTIMATE
###############################################################################

def estimate_tokens(text):
    """~4 characters per token. For the pre-flight estimate only; measured
    against a real run it comes out ~10% high, the safe direction."""

    return len(text) // 4


def print_estimate(items, dimensions):
    input_tokens = 0
    calls = 0
    for item in items:
        for dimension in dimensions:
            system_message, user_message = build_judge_prompt(
                dimension, item["document"], item["reference"],
                item["candidate_a"], item["candidate_b"], SURFACE_NOTE_ENABLED,
            )
            input_tokens += estimate_tokens(system_message) + estimate_tokens(user_message)
            calls += 1

    output_tokens = calls * EXPECTED_OUTPUT_TOKENS
    cost = price_of(MODEL, input_tokens, output_tokens)

    print(f"\nmodel                {MODEL}")
    print(f"reasoning effort     {REASONING_EFFORT}")
    print(f"surface note         {'included' if SURFACE_NOTE_ENABLED else 'OMITTED'}")
    print(f"dimensions           {', '.join(dimensions)}")
    print(f"items                {len(items)}")
    print(f"API calls            {calls}")
    print(f"input tokens  (est)  {input_tokens:,}")
    print(f"output tokens (est)  {output_tokens:,}  "
          f"({EXPECTED_OUTPUT_TOKENS}/call, incl. reasoning)")
    print(f"COST          (est)  ${cost:,.2f}")
    print("\nNothing was sent. Drop --estimate to run.")

###############################################################################
# RESUME
###############################################################################

def read_partial(path):
    """
    Already-judged pairs from an interrupted run, keyed by (item, dimension).

    A run can be killed mid-write, so a broken trailing line is dropped. Unlike
    step 7 there is no contiguity requirement: every record carries its own
    (item, dimension) key, so an arbitrary subset can be resumed.
    """

    if not path.exists():
        return {}

    done = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                break  # truncated last line
            if "item" in entry and "dimension" in entry:
                done[(entry["item"], entry["dimension"])] = entry

    return done

###############################################################################
# JUDGING RUN
###############################################################################

def run_judging(items, dimensions, out_dir, concurrency):
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / f"{BASE_NAME}.partial.jsonl"
    json_path = out_dir / f"{BASE_NAME}.json"

    done = read_partial(partial_path)
    tasks = [(item, dimension)
             for dimension in dimensions
             for item in items
             if (item["item"], dimension) not in done]

    if done:
        print(f"resuming: {len(done)} pairs already judged, {len(tasks)} to go")

    if tasks:
        load_client_if_needed()
        handle = partial_path.open("a", encoding="utf-8")
        try:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                # Submitted in chunks so results can be written in a stable
                # order; the API judges each call independently, so chunking
                # affects throughput only.
                for start in tqdm(range(0, len(tasks), concurrency),
                                  desc="judging", unit="chunk"):
                    chunk = tasks[start:start + concurrency]
                    futures = [pool.submit(judge_one_with_retry, item, dimension)
                               for item, dimension in chunk]
                    for future in futures:
                        entry = future.result()
                        done[(entry["item"], entry["dimension"])] = entry
                        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        finally:
            handle.close()

    entries = [done[(item["item"], dimension)]
               for dimension in dimensions
               for item in items
               if (item["item"], dimension) in done]

    json_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    partial_path.unlink(missing_ok=True)
    print(f"\n{len(entries)} judgements -> {json_path}")
    return entries

###############################################################################
# HUMAN VOTES
###############################################################################

VALID_VOTES = {"a": "A", "b": "B", "tie": "tie", "t": "tie"}


def load_human_answers(path):
    """
    Reads the annotators' votes.

    Expected CSV (header required, column order free, case-insensitive):

        item,annotator,faithfulness,informativeness
        1,ann1,A,B
        1,ann2,A,A
        1,ann3,tie,A
        2,ann1,B,B
        ...

    `annotator` is optional -- without it every row is treated as coming from
    the same rater. Votes are A / B / tie, exactly as written in the packet's
    brackets. Blank cells are treated as "not answered" and skipped rather than
    guessed at.
    """

    if not path.exists():
        raise SystemExit(f"human answers not found: {path}")

    votes = defaultdict(dict)          # (item, dimension) -> {annotator: A/B/tie}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{path.name} has no header row")

        columns = {name.strip().lower(): name for name in reader.fieldnames}
        if "item" not in columns:
            raise SystemExit(f"{path.name} has no 'item' column")

        present = [d for d in DIMENSIONS if d in columns]
        if not present:
            raise SystemExit(
                f"{path.name} has no dimension column; expected one of "
                f"{list(DIMENSIONS)}"
            )

        for line_no, row in enumerate(reader, start=2):
            raw_item = (row.get(columns["item"]) or "").strip()
            if not raw_item:
                continue
            try:
                item = int(raw_item)
            except ValueError:
                raise SystemExit(f"{path.name} line {line_no}: bad item {raw_item!r}")

            annotator = (row.get(columns["annotator"], "") or "").strip() \
                if "annotator" in columns else ""
            annotator = annotator or "annotator_1"

            for dimension in present:
                cell = (row.get(columns[dimension]) or "").strip().lower()
                if not cell:
                    continue
                if cell not in VALID_VOTES:
                    raise SystemExit(
                        f"{path.name} line {line_no}: bad {dimension} vote "
                        f"{cell!r} (expected A, B or tie)"
                    )
                votes[(item, dimension)][annotator] = VALID_VOTES[cell]

    return votes


def majority_vote(slot_votes):
    """
    -> the agreed slot ("A"/"B"/"tie"), or None when there is no strict
    majority.

    With three annotators and three options a 1-1-1 or 1-1 split is possible.
    Those items are excluded from the correlation and counted in the report
    rather than being folded into "tie", which would invent an agreement the
    annotators did not reach.
    """

    if not slot_votes:
        return None
    counts = Counter(slot_votes.values())
    top, n = counts.most_common(1)[0]
    return top if n * 2 > len(slot_votes) else None

###############################################################################
# CORRELATION
###############################################################################

SCORE = {"pmi": 1, "tie": 0, "rouge": -1}


def cohens_kappa(pairs):
    """Chance-corrected agreement between two raters over categorical labels."""

    n = len(pairs)
    if n == 0:
        return float("nan")

    observed = sum(1 for a, b in pairs if a == b) / n
    first = Counter(a for a, _ in pairs)
    second = Counter(b for _, b in pairs)
    expected = sum((first[c] / n) * (second[c] / n)
                   for c in set(first) | set(second))

    if expected >= 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def kendall_tau_b(pairs):
    """
    Rank correlation with a tie correction -- the right choice here because
    the scale is ordinal with only three levels (pmi > tie > rouge) and ties
    are common.
    """

    n = len(pairs)
    if n < 2:
        return float("nan")

    concordant = discordant = 0
    ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pairs[i][0] - pairs[j][0]
            dy = pairs[i][1] - pairs[j][1]
            product = dx * dy
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            else:
                if dx == 0:
                    ties_x += 1
                if dy == 0:
                    ties_y += 1

    n0 = n * (n - 1) / 2
    denominator = math.sqrt((n0 - ties_x) * (n0 - ties_y))
    if denominator == 0:
        return float("nan")
    return (concordant - discordant) / denominator


def human_agreement(votes, items_by_number, dimension):
    """
    Mean pairwise agreement among the annotators themselves -- the ceiling any
    automatic judge is being measured against. A judge that matches the humans
    as often as they match each other has nothing left to explain.
    """

    scores = []
    kappa_pairs = defaultdict(list)
    for (item, dim), slot_votes in votes.items():
        if dim != dimension or len(slot_votes) < 2:
            continue
        names = sorted(slot_votes)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = slot_votes[names[i]], slot_votes[names[j]]
                scores.append(1.0 if a == b else 0.0)
                kappa_pairs[(names[i], names[j])].append((a, b))

    if not scores:
        return None

    kappas = [cohens_kappa(p) for p in kappa_pairs.values() if len(p) > 1]
    kappas = [k for k in kappas if not math.isnan(k)]
    return {
        "agreement": sum(scores) / len(scores),
        "kappa": sum(kappas) / len(kappas) if kappas else float("nan"),
        "raters": len({name for sv in votes.values() for name in sv}),
    }


def correlation_report(entries, items_by_number, votes):
    """Builds the human-readable correlation section."""

    gpt = {(e["item"], e["dimension"]): e for e in entries}
    lines = ["", "=" * 78,
             "CORRELATION: GPT JUDGE vs. HUMAN ANNOTATORS",
             "=" * 78]

    for dimension in DIMENSIONS:
        rows = []
        no_majority = 0
        missing_gpt = 0

        for item in sorted(items_by_number):
            slot_votes = votes.get((item, dimension))
            if not slot_votes:
                continue
            slot = majority_vote(slot_votes)
            if slot is None:
                no_majority += 1
                continue

            record = gpt.get((item, dimension))
            if record is None:
                missing_gpt += 1
                continue

            meta = items_by_number[item]
            human_winner = decode_winner(slot, meta["system_a"], meta["system_b"]) \
                if slot in ("A", "B") else "tie"
            rows.append({
                "dataset": meta["dataset"],
                "human": human_winner,
                "gpt": record["gpt_winner"],
                "human_slot": slot,
                "gpt_slot": record["raw_result"],
            })

        lines += ["", "-" * 78, dimension.upper(), "-" * 78]

        if not rows:
            lines.append("  no comparable items (no human votes loaded)")
            continue

        ceiling = human_agreement(votes, items_by_number, dimension)
        if ceiling:
            lines.append(
                f"  human inter-annotator ({ceiling['raters']} raters): "
                f"agreement {ceiling['agreement']:6.1%}   "
                f"mean pairwise kappa {ceiling['kappa']:+.3f}"
            )
        if no_majority:
            lines.append(f"  excluded, no majority among annotators: {no_majority}")
        if missing_gpt:
            lines.append(f"  excluded, no GPT verdict on file: {missing_gpt}")

        def block(label, subset):
            if not subset:
                return
            agreement = sum(1 for r in subset if r["human"] == r["gpt"]) / len(subset)
            kappa = cohens_kappa([(r["human"], r["gpt"]) for r in subset])
            tau = kendall_tau_b([(SCORE[r["human"]], SCORE[r["gpt"]])
                                 for r in subset])
            human_pmi = sum(1 for r in subset if r["human"] == "pmi")
            gpt_pmi = sum(1 for r in subset if r["gpt"] == "pmi")
            human_tie = sum(1 for r in subset if r["human"] == "tie")
            gpt_tie = sum(1 for r in subset if r["gpt"] == "tie")
            lines.append(
                f"  {label:<10} n={len(subset):>3}  agree {agreement:6.1%}  "
                f"kappa {kappa:+.3f}  tau-b {tau:+.3f}  "
                f"PMI-wins h/g {human_pmi:>3}/{gpt_pmi:<3}  "
                f"ties h/g {human_tie:>3}/{gpt_tie:<3}"
            )

        lines.append("")
        block("ALL", rows)
        for dataset in sorted({r["dataset"] for r in rows}):
            block(dataset, [r for r in rows if r["dataset"] == dataset])

        # Slot-A preference, for both sides, on the identical arrangement.
        # A judge that picks slot A far more often than the humans do on the
        # same items has a positional bias, and that alone can cap the
        # correlation no matter how good its reasoning is.
        decided = [r for r in rows if r["human_slot"] in ("A", "B")
                   and r["gpt_slot"] in ("A", "B")]
        if decided:
            human_a = sum(1 for r in decided if r["human_slot"] == "A") / len(decided)
            gpt_a = sum(1 for r in decided if r["gpt_slot"] == "A") / len(decided)
            lines.append(
                f"  slot-A rate on decided pairs (n={len(decided)}): "
                f"human {human_a:6.1%}   gpt {gpt_a:6.1%}"
            )

    lines.append("")
    return "\n".join(lines)

###############################################################################
# SUMMARY LOG
###############################################################################

def summary_text(entries, items_by_number):
    lines = ["=" * 78,
             "GPT JUDGE ON THE HUMAN-EVALUATION SET",
             "=" * 78, ""]

    if entries:
        # Read back from the records, not from the current globals: a
        # --report-only pass must describe the run that produced the file.
        notes = {e.get("surface_note") for e in entries}
        note_state = ("included" if notes == {True}
                      else "OMITTED" if notes == {False}
                      else f"MIXED {notes} -- these runs should not be pooled")
        lines += [
            f"model              {entries[0].get('model')}",
            f"reasoning effort   {entries[0].get('reasoning_effort')}",
            f"surface note       {note_state}",
        ]

    total_cost = sum(e.get("cost_usd", 0.0) for e in entries)
    total_in = sum(e.get("input_tokens", 0) for e in entries)
    total_out = sum(e.get("output_tokens", 0) for e in entries)
    lines += [
        f"judgements         {len(entries)}",
        f"input tokens       {total_in:,}",
        f"output tokens      {total_out:,}",
        f"cost               ${total_cost:,.4f}",
        "",
    ]

    for dimension in DIMENSIONS:
        subset = [e for e in entries if e["dimension"] == dimension]
        if not subset:
            continue
        counts = Counter(e["gpt_winner"] for e in subset)
        unparsed = sum(1 for e in subset if e.get("raw_result") == "TIE_2")
        n = len(subset)
        lines += ["-" * 78, dimension.upper(), "-" * 78,
                  f"  n={n}   pmi {counts['pmi']:>3} ({counts['pmi']/n:5.1%})   "
                  f"rouge {counts['rouge']:>3} ({counts['rouge']/n:5.1%})   "
                  f"tie {counts['tie']:>3} ({counts['tie']/n:5.1%})"]
        if unparsed:
            lines.append(f"  no [RESULT] tag emitted: {unparsed} "
                         f"(counted as ties -- check MAX_OUTPUT_TOKENS)")

        for dataset in sorted({e["dataset"] for e in subset}):
            rows = [e for e in subset if e["dataset"] == dataset]
            c = Counter(e["gpt_winner"] for e in rows)
            lines.append(f"    {dataset:<8} n={len(rows):>3}  "
                         f"pmi {c['pmi']:>3}  rouge {c['rouge']:>3}  tie {c['tie']:>3}")
        lines.append("")

    return "\n".join(lines)

###############################################################################
# ENTRY POINT
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Correlation check between a paid GPT judge and the "
                    "105-item human evaluation set."
    )
    parser.add_argument("--estimate", action="store_true",
                        help="print the projected token count and cost, send nothing")
    parser.add_argument("--report-only", action="store_true",
                        help="skip the API and rebuild the report from the "
                             "existing results JSON")
    parser.add_argument("--human-answers", type=Path, default=None,
                        help="CSV of annotator votes; adds the correlation "
                             "section (see load_human_answers)")
    parser.add_argument("--dimensions", default=",".join(DIMENSIONS),
                        help="comma-separated subset of "
                             f"{','.join(DIMENSIONS)}")
    parser.add_argument("--eval-set", type=Path, default=EVAL_SET_PATH,
                        help="the human_eval_set.txt the annotators received")
    parser.add_argument("--answer-key", type=Path, default=ANSWER_KEY_PATH,
                        help="answer_key.txt for that packet")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="where results and the log are written")
    parser.add_argument("--model", default=MODEL, help=f"default {MODEL}")
    parser.add_argument("--reasoning-effort", default=REASONING_EFFORT,
                        choices=("none", "low", "medium", "high"))
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--no-surface-note", action="store_true",
                        help="drop the 'ignore generation artifacts' note from "
                             "the prompt, leaving it at step 7's shape. Write "
                             "the run to its own --out-dir so it is not mixed "
                             "with a run that had the note.")
    return parser.parse_args()


def main():
    global MODEL, REASONING_EFFORT, SURFACE_NOTE_ENABLED

    args = parse_args()
    MODEL = args.model
    REASONING_EFFORT = args.reasoning_effort
    SURFACE_NOTE_ENABLED = not args.no_surface_note

    dimensions = [d.strip() for d in args.dimensions.split(",") if d.strip()]
    unknown = [d for d in dimensions if d not in DIMENSIONS]
    if unknown:
        raise SystemExit(f"unknown dimension(s): {unknown}; "
                         f"expected from {list(DIMENSIONS)}")

    items = build_items(args.eval_set, args.answer_key)
    items_by_number = {item["item"]: item for item in items}
    print(f"{len(items)} items loaded from {args.eval_set}")

    if args.estimate:
        print_estimate(items, dimensions)
        return

    json_path = args.out_dir / f"{BASE_NAME}.json"
    if args.report_only:
        if not json_path.exists():
            raise SystemExit(f"nothing to report on: {json_path} does not exist")
        entries = json.loads(json_path.read_text(encoding="utf-8"))
        print(f"{len(entries)} judgements read from {json_path}")
    else:
        entries = run_judging(items, dimensions, args.out_dir, args.concurrency)

    report = summary_text(entries, items_by_number)

    if args.human_answers:
        votes = load_human_answers(args.human_answers)
        report += correlation_report(entries, items_by_number, votes)
    else:
        report += ("\nNo --human-answers given, so no correlation was computed.\n"
                   "Re-run with --report-only --human-answers votes.csv once the\n"
                   "annotators are finished; it costs nothing.\n")

    log_path = args.out_dir / f"{BASE_NAME}.log"
    log_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"log -> {log_path}")


if __name__ == "__main__":
    main()

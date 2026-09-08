"""
CORRELATION CHECK -- paid GPT judge vs. the 105-item HUMAN EVALUATION set.

The human study (see "HUMAN Eval/prepare_human_eval_set.py") shows three
annotators 105 items -- 35 each from cnn, wikihow and xsum, all at the 8M
checkpoint -- and asks two separate questions per item:

    BEST ACCORDING TO FAITHFULNESS      (A / B / tie)
    BEST ACCORDING TO INFORMATIVENESS   (A / B / tie)

This script puts the SAME 105 items, in the SAME candidate arrangement, to
gpt-5.6-sol at reasoning effort "medium", one question at a time: 105
faithfulness calls + 105 informativeness calls = 210 paid calls. It then
measures how well the paid judge agrees with the humans, three ways:

    1. judge vs. the annotator CONSENSUS  -- the headline number
    2. judge vs. EACH annotator on their own
    3. every rater against every other, judge included -- who agrees with
       whom, as raw counts and percentages

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

So this is ONE draw from the judge, and the agreement figures below carry an
unknown amount of sampling noise. Nothing here estimates that noise. If a
later comparison turns on a small difference between two configurations, that
is the point at which repeating a run and looking at the spread would start to
matter; for a single headline number it is not worth the money.

COST
----
210 calls over ~105 source documents (~55k words, read twice). At
effort="medium" on gpt-5.6-sol that is roughly 280k input + 105k output
tokens -- much of the output being reasoning -- or about $4.55. --estimate
still exists and still sends nothing; RUN IT FIRST -- at Sol prices a mistake
in the prompts or the packet costs real money.

That $4.55 is two multipliers on the ~$0.03 a single run of this file
originally cost, and either can be given back on its own:

  model    Sol is 25x Luna per token ($5.00/$30.00 vs $0.20/$1.20 per 1M),
           and is the whole reason this is dollars rather than cents.
           --model gpt-5.6-luna brings the run to ~$0.18.
  effort   "medium" spends roughly 2.4x the output tokens of "low", nearly
           all of it reasoning; "high" would be ~5x and put the run at
           ~$8.33. The output-token budget follows the effort on its own
           (MAX_OUTPUT_TOKENS_BY_EFFORT), so changing one changes both.

Both multipliers are bets on the same thing -- that a more capable judge,
thinking longer, tracks the annotators more closely. Neither is measured yet
on this set. The honest way to find out is to run the cheap configuration
too, into its own --out-dir, and compare the two against the same human
votes; that costs ~$0.18 on top and tells you whether the $4.55 bought
anything.

SETUP
-----
  1. pip install openai
  2. export OPENAI_API_KEY="sk-..."      (never hard-code it here)
  3. copy human_eval_set.txt (the FINAL_VERSION copy) and answer_key.txt next to
     this script -- see EVAL_SET_PATH / ANSWER_KEY_PATH
  4. python GPT_API_corralation_check_with_human_eval.py --estimate
  5. python GPT_API_corralation_check_with_human_eval.py --get-gpt-results-only
  6. once the annotators are done:
     python prepare_human_answers_csv.py
     python GPT_API_corralation_check_with_human_eval.py \
            --calc-correlation-statistics-only \
            --human-answers human_answers.csv

Steps 5 and 6 are the paid half and the free half, and they are deliberately
separable -- see GET_GPT_RESULTS_ONLY / CALC_CORRELATION_STATISTICS_ONLY at
the top of the configuration. Running with neither flag does both at once,
which is what the file did before they existed.

The verdicts land in gpt_judge_vs_human_eval.json and the report in
gpt_judge_vs_human_eval.log (a judge-only run writes
gpt_judge_vs_human_eval.gpt_only.log instead, so it cannot overwrite a full
report). Step 6 sends nothing and can be re-run as often as you like -- as
more annotators come in, or as a statistic is added -- without paying again.

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

At the default (Sol, medium) each side of that comparison is ~$4.55, so settle
it on the cheap configuration instead -- --model gpt-5.6-luna
--reasoning-effort low makes the pair ~$0.07, and there is no reason to think
a prompt sentence that matters on one model stops mattering on the other.
Every record stores which way it was run, and the log says so at the top.
"""

import argparse
import csv
import json
import math
import os
import re
import textwrap
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
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

# THE TWO HALVES OF THIS SCRIPT, AND HOW TO RUN JUST ONE OF THEM
#
# A full run does two quite different jobs back to back:
#
#   1. ask the judge  -- 210 paid API calls, minutes, costs ~$4.55, and the
#                        only part that can fail on a network or a bill;
#   2. do the sums    -- consensus, correlation, agreement tables. Free,
#                        instant, and the part you re-run over and over as
#                        annotators come in or a statistic gets added.
#
# Tying them together means every look at the numbers drags the paid half
# behind it, so either half can be run alone. Set ONE of these to True (or
# pass the matching --flag, which overrides whatever is set here):
#
#   GET_GPT_RESULTS_ONLY             judge, save the JSON, stop. Human votes
#                                    are not needed and are not read.
#   CALC_CORRELATION_STATISTICS_ONLY read the saved JSON, compute everything,
#                                    send nothing. Needs --human-answers.
#
# Both False (the default) runs the whole thing end to end, as before. Both
# True is a contradiction and is refused rather than silently resolved.
#
# Judging is idempotent either way: verdicts already in the JSON or the
# .partial.jsonl are never bought twice, so re-running the first half after
# it has finished costs nothing.
GET_GPT_RESULTS_ONLY = False              # MODIFY HERE TO ONLY CALL THE API
CALC_CORRELATION_STATISTICS_ONLY = False  # MODIFY HERE TO ONLY DO THE SUMS

# Sol -- the most capable judge available, chosen deliberately here and only
# here. Steps 5/6/7 run Luna because they issue judgements by the hundred
# thousand and Sol is 25x the price per token; this file issues 630, so the
# whole grid's reasoning about cost per judgement does not apply to it. What
# is being bought is agreement with the annotators on 105 items, and that is
# the one number in the project no cheaper model can be substituted into
# after the fact.
#
# Still pinned to the exact model id, NOT the "gpt-5.6" alias, for step 7's
# reason: the alias is OpenAI's to repoint without notice, which would
# silently change what a run means. Sol here is a choice, not an alias
# resolving to one.
#
# NOTE the price gap before re-running anything: at the default effort a run
# costs ~$4.55 on Sol against ~$0.18 on Luna. --model gpt-5.6-luna switches
# back, and the model is recorded per call so the two can never be pooled by
# accident.
MODEL = "gpt-5.6-sol"

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

# "medium" -- a deliberate compromise, and the place where the cost of this
# file is actually decided.
#
# Effort is the one knob that measurably moved this judge: going from "none"
# to "low" halved its slot-A preference (+25.1 -> +10.9 points on wikihow/1M)
# in step 7 -- more deliberation bought less positional bias. A correlation
# study is exactly where that bias does the most damage, since it caps the
# achievable agreement however good the judgement itself is, which argues for
# buying as much deliberation as possible.
#
# Against that: reasoning tokens are billed at Sol's output rate, and they
# dominate the bill. A run comes to ~$4.55 at "medium" against ~$8.33 at
# "high" -- the same 210 verdicts for $3.78 less. "medium" is the setting that
# keeps most of the measured benefit of deliberating at all (the big step in
# step 7 was none -> low, not low -> high) without paying the top rate for
# reasoning nobody has yet shown is needed on 105 items.
#
# Nothing here is measured on THIS set, so treat it as a starting point, not
# a finding: --reasoning-effort high is one flag away if the correlation at
# medium looks capped by the judge rather than by the task, and
# --reasoning-effort low reproduces step 7's original setting exactly. Every
# record stores the effort it was judged at, so runs at different efforts can
# never be pooled by accident.
REASONING_EFFORT = "medium"

# Room for the feedback and the [RESULT] tag -- and, on a reasoning model, for
# the reasoning tokens too, which are billed and counted against this same
# budget. A call that exhausts the budget mid-reasoning returns no [RESULT] tag
# at all and is scored as a tie, so the budget has to scale with the effort:
# step 7's 768 was measured at effort="low" (~209 output tokens per call) and
# would truncate a meaningful share of calls at "medium" or above.
MAX_OUTPUT_TOKENS_BY_EFFORT = {
    "none":   512,
    "low":    768,
    "medium": 1536,
    "high":   3072,
}
DEFAULT_MAX_OUTPUT_TOKENS = 3072

# Parallel in-flight requests. Purely a throughput knob -- each call is judged
# independently, so concurrency cannot change a verdict.
CONCURRENCY = 8

# Retry policy for 429s and transient 5xx. Sleeps 2, 4, 8, 16, 32 seconds.
MAX_RETRIES = 5

# Output tokens per verdict, for the --estimate projection only, by effort.
# "low" is step 7's measured figure (209 = ~117 reasoning + ~92 visible); the
# others are projected from it, reasoning tokens being what actually scales
# with effort. They lean high, which is the safe direction for a cost estimate.
#
# These are the softest numbers in the file, and on Sol they carry most of the
# bill: output is ~69% of the projected $4.55, and the 500 below is a guess
# extrapolated from a DIFFERENT model at a DIFFERENT effort. Sol has never
# been measured on this prompt. The projection cannot run away without limit
# -- MAX_OUTPUT_TOKENS_BY_EFFORT caps each call at 1536 at this effort, so the
# true worst case is ~$11 rather than $4.55 -- but that is a wide bracket to
# start a paid run inside.
#
# So bound it before committing: start the run, Ctrl-C after a chunk or two,
# read the real output_tokens back out of
# gpt_judge_vs_human_eval.partial.jsonl, correct the number here, and re-run
# --estimate. The partial is a resume cache, so those calls are not wasted --
# the full run picks up from them.
EXPECTED_OUTPUT_TOKENS_BY_EFFORT = {
    "none":   110,
    "low":    209,
    "medium": 500,
    "high":   1100,
}
DEFAULT_EXPECTED_OUTPUT_TOKENS = 1100

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


def max_output_tokens_for(model):
    """Output budget for one call. On a reasoning model it has to cover the
    reasoning tokens as well, so it follows the effort; a non-reasoning model
    spends none and takes the smallest budget."""

    if not is_reasoning_model(model):
        return MAX_OUTPUT_TOKENS_BY_EFFORT["none"]
    return MAX_OUTPUT_TOKENS_BY_EFFORT.get(
        REASONING_EFFORT, DEFAULT_MAX_OUTPUT_TOKENS)


def expected_output_tokens_for(model):
    """Projected output tokens per call, for --estimate only."""

    if not is_reasoning_model(model):
        return EXPECTED_OUTPUT_TOKENS_BY_EFFORT["none"]
    return EXPECTED_OUTPUT_TOKENS_BY_EFFORT.get(
        REASONING_EFFORT, DEFAULT_EXPECTED_OUTPUT_TOKENS)


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
# --no-surface-note is for. Settle it by measurement rather than by argument:
# run both into separate --out-dir folders and compare. On the default (Sol,
# medium) that pair is ~$9, so run the comparison on --model gpt-5.6-luna
# --reasoning-effort low instead, where it is ~$0.07 and answers the same
# question about the prompt. Every record stores which way it was run, so the
# two cannot be confused.
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
    are reported separately, so it is obvious if the output budget is cutting
    the judge off -- the failure mode to watch for at any effort above "none",
    where the reasoning tokens come out of the same budget. Carried over from
    step 5/7.
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
        "max_output_tokens": max_output_tokens_for(MODEL),
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

    per_call_output = expected_output_tokens_for(MODEL)
    output_tokens = calls * per_call_output
    cost = price_of(MODEL, input_tokens, output_tokens)

    print(f"\nmodel                {MODEL}")
    print(f"reasoning effort     {REASONING_EFFORT}")
    print(f"output budget/call   {max_output_tokens_for(MODEL):,} tokens")
    print(f"surface note         {'included' if SURFACE_NOTE_ENABLED else 'OMITTED'}")
    print(f"dimensions           {', '.join(dimensions)}")
    print(f"items                {len(items)}")
    print(f"API calls            {calls:,}")
    print(f"input tokens  (est)  {input_tokens:,}")
    print(f"output tokens (est)  {output_tokens:,}  "
          f"({per_call_output}/call, incl. reasoning)")
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

    # A finished run leaves its verdicts in the JSON and deletes the partial,
    # so without this a second run would re-buy all 210 calls. Seeded per
    # (item, dimension) rather than skipping wholesale, so adding a dimension
    # later pays only for the new one.
    if json_path.exists():
        try:
            for entry in json.loads(json_path.read_text(encoding="utf-8")):
                done.setdefault((entry["item"], entry["dimension"]), entry)
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"[WARN] {json_path.name} is unreadable and will be "
                  f"rebuilt from scratch")
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


def consensus_vote(slot_votes):
    """
    The annotators' agreed slot -- "A", "B" or "tie". None only when nobody
    voted on the item at all.

    The rule is plurality, with a split that has no plurality resolved to
    "tie". Worked through for three annotators, which is every case that can
    arise:

        A, A, A     -> A     unanimous
        A, A, B     -> A     2-1
        A, B, B     -> B     2-1
        A, A, tie   -> A     2-1
        A, tie, tie -> tie   2-1
        A, B, tie   -> tie   1-1-1, nothing leads

    The last line is the only judgement call in here, and it is the reason
    this is not just Counter.most_common(1). Three annotators who each say
    something different have told you the item does not separate the two
    systems, which is what "tie" means on this scale -- so it is recorded as
    a tie rather than dropped.

    Dropping was the previous behaviour and it was worse in a specific way:
    excluded items are not missing at random. They are the hardest items, the
    ones the annotators split on, and removing them inflates every agreement
    figure computed afterwards -- including the judge's -- by quietly deleting
    the cases most likely to be got wrong. Folding them to "tie" keeps n fixed
    at 105 and keeps the hard items in the denominator.

    Note this is a rule about the CONSENSUS only. Each annotator's own votes
    are reported separately and untouched, so nothing here can hide behind it.

    With an even number of raters a 1-1 or 2-2 split also has no plurality and
    likewise becomes "tie", by the same reasoning.
    """

    if not slot_votes:
        return None

    counts = Counter(slot_votes.values())
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return "tie"          # no single leader
    return ranked[0][0]

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


def comparison_rows(entries, items_by_number, votes, dimension, human_slot_of):
    """
    The comparable (human, GPT) pairs for one dimension.

    `human_slot_of` turns one item's {annotator: slot} into the single human
    slot to compare against -- consensus_vote for the consensus block, or
    "just this annotator's vote" for the per-annotator blocks. Returning None
    skips the item, which is how an annotator who did not answer everything is
    handled without affecting anyone else's n.

    Both sides are decoded from slots to systems here, through this item's own
    answer-key row, so "A" is never compared against "A" -- it is PMI against
    PMI.
    """

    gpt = {(e["item"], e["dimension"]): e for e in entries}
    rows = []
    no_human = 0
    missing_gpt = 0

    for item in sorted(items_by_number):
        slot_votes = votes.get((item, dimension))
        human_slot = human_slot_of(slot_votes) if slot_votes else None
        if human_slot is None:
            no_human += 1
            continue

        record = gpt.get((item, dimension))
        if record is None:
            missing_gpt += 1
            continue

        meta = items_by_number[item]
        human_winner = decode_winner(human_slot, meta["system_a"], meta["system_b"]) \
            if human_slot in ("A", "B") else "tie"
        rows.append({
            "dataset": meta["dataset"],
            "human": human_winner,
            "gpt": record["gpt_winner"],
            "human_slot": human_slot,
            "gpt_slot": record["raw_result"],
        })

    return rows, no_human, missing_gpt


def block_stats(subset):
    """The numbers behind one line of a correlation table."""

    if not subset:
        return None
    return {
        "n": len(subset),
        "agreement": sum(1 for r in subset if r["human"] == r["gpt"]) / len(subset),
        "kappa": cohens_kappa([(r["human"], r["gpt"]) for r in subset]),
        "tau": kendall_tau_b([(SCORE[r["human"]], SCORE[r["gpt"]]) for r in subset]),
        "human_pmi": sum(1 for r in subset if r["human"] == "pmi"),
        "gpt_pmi": sum(1 for r in subset if r["gpt"] == "pmi"),
        "human_tie": sum(1 for r in subset if r["human"] == "tie"),
        "gpt_tie": sum(1 for r in subset if r["gpt"] == "tie"),
    }


def stats_line(label, block):
    return (f"  {label:<12} n={block['n']:>3}  agree {block['agreement']:6.1%}  "
            f"kappa {block['kappa']:+.3f}  tau-b {block['tau']:+.3f}  "
            f"PMI-wins h/g {block['human_pmi']:>3}/{block['gpt_pmi']:<3}  "
            f"ties h/g {block['human_tie']:>3}/{block['gpt_tie']:<3}")


def slot_a_line(rows):
    """
    Slot-A preference, for both sides, on the identical arrangement.

    A judge that picks slot A far more often than the humans do on the same
    items has a positional bias, and that alone can cap the correlation no
    matter how good its reasoning is.
    """

    decided = [r for r in rows
               if r["human_slot"] in ("A", "B") and r["gpt_slot"] in ("A", "B")]
    if not decided:
        return None
    human_a = sum(1 for r in decided if r["human_slot"] == "A") / len(decided)
    gpt_a = sum(1 for r in decided if r["gpt_slot"] == "A") / len(decided)
    return (f"  slot-A rate on decided pairs (n={len(decided)}): "
            f"human {human_a:6.1%}   gpt {gpt_a:6.1%}")


def annotators_in(votes):
    """Every annotator id that appears anywhere in the votes, sorted."""

    return sorted({name for slot_votes in votes.values() for name in slot_votes})


def correlation_report(entries, items_by_number, votes):
    """
    The judge against the humans, two ways per dimension:

      1. against the annotator CONSENSUS -- the headline number, and the one
         to quote, because it is the humans' collective answer;
      2. against EACH annotator separately -- which says whether a weak
         consensus figure is the judge disagreeing with everyone, or the
         judge siding with one annotator against the others.

    Both are broken down by dataset, since cnn / wikihow / xsum behave quite
    differently and a pooled number hides that.
    """

    lines = ["", "=" * 78,
             "CORRELATION: GPT JUDGE vs. HUMAN ANNOTATORS",
             "=" * 78]

    names = annotators_in(votes)

    for dimension in DIMENSIONS:
        lines += ["", "-" * 78, dimension.upper(), "-" * 78]

        rows, no_human, missing_gpt = comparison_rows(
            entries, items_by_number, votes, dimension, consensus_vote)

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
        if no_human:
            lines.append(f"  excluded, no human vote on file: {no_human}")
        if missing_gpt:
            lines.append(f"  excluded, no GPT verdict on file: {missing_gpt}")

        lines += ["", "  vs. ANNOTATOR CONSENSUS"]
        lines.append(stats_line("ALL", block_stats(rows)))
        for dataset in sorted({r["dataset"] for r in rows}):
            lines.append(stats_line(
                dataset, block_stats([r for r in rows if r["dataset"] == dataset])))
        slot_a = slot_a_line(rows)
        if slot_a:
            lines.append(slot_a)

        if len(names) > 1:
            lines += ["", "  vs. EACH ANNOTATOR (same judge, one rater at a time)"]
            for name in names:
                own_rows, _, _ = comparison_rows(
                    entries, items_by_number, votes, dimension,
                    lambda slot_votes, name=name: slot_votes.get(name))
                block = block_stats(own_rows)
                if block is None:
                    lines.append(f"  {name:<12} no votes on this dimension")
                    continue
                lines.append(stats_line(name, block))

    lines.append("")
    return "\n".join(lines)

###############################################################################
# WHO AGREES WITH WHOM
###############################################################################

# The correlation section above measures the judge against the humans. This one
# measures everybody against everybody -- each pair of annotators, all three
# together, and the same combinations with the judge substituted in or added.
#
# It is reported as raw counts as well as percentages on purpose: with 105 items
# the difference between 61 and 66 agreements is a percentage point and a half,
# and the count makes it obvious how few items that actually is.
#
# Agreement is computed on the A/B/tie SLOTS rather than on decoded systems. The
# two give identical numbers -- decoding is a per-item relabelling of the same
# three categories, so two raters match on slots exactly when they match on
# systems -- and slots keep the judge's unparsed "TIE_2" verdicts in one place.

GPT_LABEL = "GPT"


def gpt_label_for(names):
    """A label for the judge that cannot collide with an annotator id."""

    label = GPT_LABEL
    while label in names:
        label += "_"
    return label


def rater_slots(entries, votes, dimension, judge_label):
    """-> {rater: {item: slot}} across the annotators and the judge."""

    slots = defaultdict(dict)

    for (item, dim), slot_votes in votes.items():
        if dim != dimension:
            continue
        for name, slot in slot_votes.items():
            slots[name][item] = slot

    for entry in entries:
        if entry["dimension"] != dimension:
            continue
        # "TIE_2" (no [RESULT] tag emitted) is scored as a tie here, exactly as
        # it is everywhere else in this file.
        raw = entry["raw_result"]
        slots[judge_label][entry["item"]] = raw if raw in ("A", "B") else "tie"

    return slots


def group_agreement(slots, group):
    """
    -> (items where every member of `group` voted, items where they all gave
    the same answer).

    Unanimity, not pairwise-averaged: for a group of three this is "all three
    said the same thing", which is the number people mean by "all three agreed".
    """

    shared = set.intersection(*(set(slots[name]) for name in group))
    same = sum(1 for item in shared
               if len({slots[name][item] for name in group}) == 1)
    return len(shared), same


def mean_of(values):
    """Plain mean over the usable values, NaNs dropped. NaN if none survive."""

    usable = [v for v in values if v is not None and not math.isnan(v)]
    if not usable:
        return float("nan")
    return sum(usable) / len(usable)


def mean_rows(label, rows, width=34):
    """
    One averaged line over a set of agreement rows.

    The average is UNWEIGHTED -- the mean of the percentages actually printed
    above it, which is what "average of these rows" means. When every row
    covers the same items, which is the normal case here (105 items, everyone
    answering everything), that is identical to pooling the counts. When it is
    not -- an annotator who skipped items -- the two differ, so the pooled
    figure is printed alongside rather than silently chosen for you.
    """

    if len(rows) < 2:
        return []                       # a "mean" of one row is that row

    mean_pct = mean_of([r["pct"] for r in rows])
    kappas = [r["kappa"] for r in rows if r["kappa"] is not None]
    line = f"    {label:<{width}} {'':>7}  {mean_pct:6.1%}"
    if kappas:
        line += f"   kappa {mean_of(kappas):+.3f}"

    sizes = {r["shared"] for r in rows}
    if len(sizes) > 1:
        same = sum(r["same"] for r in rows)
        shared = sum(r["shared"] for r in rows)
        line += f"   (pooled {same}/{shared} = {same / shared:.1%})"

    return [line]


def rater_agreement_report(entries, items_by_number, votes):
    """Every combination of raters, from pairs upward, then their averages."""

    names = annotators_in(votes)
    if not names:
        return ""

    judge_label = gpt_label_for(names)
    everyone = names + [judge_label]

    lines = ["", "=" * 78,
             "AGREEMENT AMONG RATERS (annotators and the GPT judge)",
             "=" * 78,
             "",
             "Raw agreement: how often two or more raters wrote the same answer.",
             "Unanimous for groups of three or more. Percentages are over the",
             "items every member of that group answered, which is why n can",
             "differ between rows.",
             "",
             "Cohen's kappa is shown for pairs only -- it is a two-rater",
             "statistic. It corrects for agreement expected by chance, so on a",
             "three-category scale where everyone ties often, a high raw",
             "percentage can still be a low kappa.",
             "",
             "Each block ends with the averages of its own rows, split three",
             "ways: the annotators among themselves (the human ceiling), the",
             "judge against the annotators (what the judge achieves against a",
             "typical single human), and everything pooled together. The first",
             "two are the pair worth comparing -- the judge is doing well when",
             "its row approaches the annotators' row, not when it approaches",
             "100%."]

    for dimension in DIMENSIONS:
        slots = rater_slots(entries, votes, dimension, judge_label)
        present = [name for name in everyone if slots.get(name)]
        if len(present) < 2:
            continue

        lines += ["", "-" * 78, dimension.upper(), "-" * 78]

        for size in range(2, len(present) + 1):
            groups = list(combinations(present, size))
            if not groups:
                continue

            if size == 2:
                heading = "  PAIRS"
            elif size == len(everyone):
                heading = f"  ALL {size} RATERS"
            else:
                heading = f"  GROUPS OF {size}"
            lines += ["", heading]

            # Annotator-only combinations first: those are the inter-annotator
            # numbers, and the judge's rows are a separate question.
            rows = []
            for group in sorted(groups, key=lambda g: (judge_label in g, g)):
                shared, same = group_agreement(slots, group)
                if not shared:
                    continue

                kappa = None
                if size == 2:
                    first, second = group
                    items = sorted(set(slots[first]) & set(slots[second]))
                    value = cohens_kappa(
                        [(slots[first][i], slots[second][i]) for i in items])
                    kappa = None if math.isnan(value) else value

                rows.append({"group": group, "shared": shared, "same": same,
                             "pct": same / shared, "kappa": kappa,
                             "has_gpt": judge_label in group})

            for row in rows:
                line = (f"    {' & '.join(row['group']):<34} "
                        f"{row['same']:>3}/{row['shared']:<3}  {row['pct']:6.1%}")
                if row["kappa"] is not None:
                    line += f"   kappa {row['kappa']:+.3f}"
                lines.append(line)

            human_rows = [r for r in rows if not r["has_gpt"]]
            gpt_rows = [r for r in rows if r["has_gpt"]]
            averages = (
                mean_rows(f"mean, annotators only ({len(human_rows)})", human_rows)
                + mean_rows(f"mean, including {judge_label} ({len(gpt_rows)})",
                            gpt_rows)
                + mean_rows(f"mean, all ({len(rows)})", rows)
            )
            if averages:
                lines.append("    " + "-" * 70)
                lines += averages

    lines.append("")
    return "\n".join(lines)

###############################################################################
# STRONG DISAGREEMENTS
###############################################################################

# Not every disagreement is worth reading back. Two raters who split A against
# "tie" have essentially the same reading of the item, one of them just held it
# to a stricter standard -- there is usually nothing to see. Two raters who
# split A against B have read the same item in opposite directions, and one of
# them is wrong in a way that is worth looking at.
#
# So only the second kind is listed, per the examples this was specified from:
#
#     rater 1: A, tie      rater 2: tie, B      -> soft, not listed
#         faithfulness A vs tie, informativeness tie vs B: neither dimension
#         has the two of them picking opposite candidates.
#
#     rater 1: A, A        rater 2: B, tie      -> STRONG, listed
#         faithfulness is A vs B. (Informativeness, A vs tie, is soft -- but one
#         dimension is enough to make the item worth a look.)
#
# Slots, not systems, but the distinction does not matter here: A vs B is
# exactly "one picked PMI and the other picked ROUGE", whichever way round the
# arrangement put them for that item.


def wrap_numbers(numbers, indent):
    """Item indices as wrapped, indented lines -- 105 of them do not fit on
    one."""

    if not numbers:
        return []
    text = ", ".join(str(n) for n in numbers)
    return [indent + line
            for line in textwrap.wrap(text, width=78 - len(indent))]


def opposite_picks(slots_a, slots_b):
    """Items where one rater said A and the other said B."""

    return sorted(item for item in set(slots_a) & set(slots_b)
                  if {slots_a[item], slots_b[item]} == {"A", "B"})


def strong_disagreement_report(entries, items_by_number, votes):
    """Which items two raters read in opposite directions, and where."""

    names = annotators_in(votes)
    if not names:
        return ""

    judge_label = gpt_label_for(names)
    everyone = names + [judge_label]
    by_dimension = {dimension: rater_slots(entries, votes, dimension, judge_label)
                    for dimension in DIMENSIONS}

    lines = ["", "=" * 78,
             "STRONG DISAGREEMENTS -- one rater picked A, the other picked B",
             "=" * 78,
             "",
             "Items read in opposite directions. A against 'tie' is a soft",
             "disagreement and is not listed; only A against B is, on the same",
             "item and the same dimension.",
             "",
             "'either' is the union: items strongly disagreed on for at least one",
             "of the two dimensions. Percentages are over the items both raters",
             "answered."]

    present = [name for name in everyone
               if any(by_dimension[d].get(name) for d in DIMENSIONS)]

    for first, second in sorted(combinations(present, 2),
                                key=lambda g: (judge_label in g, g)):
        per_dimension = {}
        shared_items = set()
        for dimension in DIMENSIONS:
            slots = by_dimension[dimension]
            slots_a = slots.get(first, {})
            slots_b = slots.get(second, {})
            shared = set(slots_a) & set(slots_b)
            if not shared:
                continue
            shared_items |= shared
            per_dimension[dimension] = (opposite_picks(slots_a, slots_b), shared)

        if not per_dimension:
            continue

        lines += ["", f"  {first} & {second}"]

        either = set()
        for dimension, (found, shared) in per_dimension.items():
            either |= set(found)
            lines.append(f"    {dimension:<16} {len(found):>3}/{len(shared):<3} "
                         f"{len(found) / len(shared):6.1%}")
            lines += wrap_numbers(found, " " * 6)

        if len(per_dimension) > 1:
            lines.append(f"    {'either':<16} {len(either):>3}/"
                         f"{len(shared_items):<3} "
                         f"{len(either) / len(shared_items):6.1%}")
            lines += wrap_numbers(sorted(either), " " * 6)

    # The union over the human pairs only: the items the annotators themselves
    # could not settle, which is the list to re-read when deciding whether an
    # item was ambiguous, mis-specified, or simply hard.
    contested = set()
    for first, second in combinations(names, 2):
        for dimension in DIMENSIONS:
            slots = by_dimension[dimension]
            contested |= set(opposite_picks(slots.get(first, {}),
                                            slots.get(second, {})))

    if len(names) > 1:
        lines += ["", "-" * 78,
                  f"  CONTESTED ITEMS -- any annotator pair, any dimension: "
                  f"{len(contested)}/{len(items_by_number)} "
                  f"({len(contested) / len(items_by_number):.1%})",
                  "-" * 78]
        lines += wrap_numbers(sorted(contested), "    ")

    lines.append("")
    return "\n".join(lines)

###############################################################################
# SUMMARY LOG
###############################################################################

def slot_a_counts(records):
    """
    The judge's own slot preference: (verdicts that picked a side, of which A).

    Ties express no position and are excluded, TIE_2 (no [RESULT] tag emitted)
    with them -- a call that ran out of budget mid-reasoning did not choose a
    slot either. The denominator is therefore smaller than n, and is printed.
    """

    decided = [e for e in records if e.get("raw_result") in ("A", "B")]
    return len(decided), sum(1 for e in decided if e["raw_result"] == "A")


def arrangement_line(items_by_number):
    """How often PMI sat in slot A, which is what the slot-A rate is read
    against: the packet is near-balanced, so a judge with no positional
    preference and no real preference lands near 50% on both."""

    total = len(items_by_number)
    if not total:
        return None
    pmi_a = sum(1 for item in items_by_number.values()
                if item["system_a"].upper() == "PMI")
    return (f"arrangement        PMI in slot A on {pmi_a} of {total} items "
            f"({pmi_a / total:.1%})")


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
    total_reasoning = sum(e.get("reasoning_tokens", 0) for e in entries)
    lines += [
        f"judgements         {len(entries)}",
        f"input tokens       {total_in:,}",
        f"output tokens      {total_out:,}  "
        f"(of which reasoning {total_reasoning:,})",
        f"cost               ${total_cost:,.4f}",
    ]
    arrangement = arrangement_line(items_by_number)
    if arrangement:
        lines.append(arrangement)
    lines.append("")

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
            lines.append(f"  no [RESULT] tag emitted: {unparsed} (counted as ties "
                         f"-- raise MAX_OUTPUT_TOKENS_BY_EFFORT for this effort)")

        for dataset in sorted({e["dataset"] for e in subset}):
            rows = [e for e in subset if e["dataset"] == dataset]
            c = Counter(e["gpt_winner"] for e in rows)
            lines.append(f"    {dataset:<8} n={len(rows):>3}  "
                         f"pmi {c['pmi']:>3}  rouge {c['rouge']:>3}  tie {c['tie']:>3}")

        # POSITIONAL BIAS, from the verdicts alone.
        #
        # slot_a_line() in the correlation report measures the same thing
        # comparatively -- judge against the humans on the identical
        # arrangement -- and needs the human votes, so it cannot run until the
        # annotators are in. This one needs nothing but the JSON, which means
        # the bias is readable straight after the paid run.
        #
        # Read it against 50%, not against the humans: the packet puts PMI in
        # slot A on 52 of 105 items, so a judge with no positional preference
        # should pick A about half the time. The two denominators differ on
        # purpose -- here it is every verdict that picked a side, there only
        # the items where the human picked a side too.
        decided, slot_a = slot_a_counts(subset)
        if decided:
            lines.append(f"  slot-A rate (of the {decided} verdicts that "
                         f"picked a side): {slot_a}/{decided} = "
                         f"{slot_a / decided:5.1%}   [50% = no preference]")
            for dataset in sorted({e["dataset"] for e in subset}):
                d, a = slot_a_counts([e for e in subset
                                      if e["dataset"] == dataset])
                if d:
                    lines.append(f"    {dataset:<8} n={d:>3}  "
                                 f"slot A {a:>3} ({a / d:5.1%})")
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
    parser.add_argument("--get-gpt-results-only", action="store_true",
                        default=GET_GPT_RESULTS_ONLY,
                        help="call the API, save the verdicts, stop before "
                             "the correlation statistics "
                             f"(default {GET_GPT_RESULTS_ONLY})")
    # --report-only is the original name for this and still works.
    parser.add_argument("--calc-correlation-statistics-only", "--report-only",
                        action="store_true",
                        default=CALC_CORRELATION_STATISTICS_ONLY,
                        help="skip the API and compute the statistics from "
                             "the existing results JSON; needs "
                             "--human-answers "
                             f"(default {CALC_CORRELATION_STATISTICS_ONLY})")
    parser.add_argument("--human-answers", type=Path, default=None,
                        help="CSV of annotator votes, as written by "
                             "prepare_human_answers_csv.py; adds the "
                             "correlation and agreement sections")
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
                        choices=("none", "low", "medium", "high"),
                        help=f"default {REASONING_EFFORT}")
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

    ask_the_judge = not args.calc_correlation_statistics_only
    do_the_sums = not args.get_gpt_results_only
    if not ask_the_judge and not do_the_sums:
        raise SystemExit(
            "GET_GPT_RESULTS_ONLY and CALC_CORRELATION_STATISTICS_ONLY are "
            "both set.\nThey are the two halves of the run, so setting both "
            "leaves nothing to do.\nSet one, or neither to run the whole "
            "thing."
        )

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

    if ask_the_judge:
        entries = run_judging(items, dimensions, args.out_dir, args.concurrency)
    else:
        if not json_path.exists():
            raise SystemExit(
                f"nothing to compute from: {json_path} does not exist.\n"
                f"Run the judging half first (GET_GPT_RESULTS_ONLY, or no "
                f"flag at all)."
            )
        entries = json.loads(json_path.read_text(encoding="utf-8"))
        print(f"{len(entries)} judgements read from {json_path}")

    report = summary_text(entries, items_by_number)

    if not do_the_sums:
        if args.human_answers:
            print("[INFO] --human-answers ignored: this run stops after the "
                  "GPT verdicts")
        report += ("\nStopped after the GPT verdicts (GET_GPT_RESULTS_ONLY).\n"
                   "The correlation and agreement statistics come from a "
                   "second, free run:\n"
                   "  python GPT_API_corralation_check_with_human_eval.py \\\n"
                   "         --calc-correlation-statistics-only \\\n"
                   "         --human-answers human_answers.csv\n")
        # Its own file, so stopping early cannot overwrite a full report that
        # is already on disk.
        log_path = args.out_dir / f"{BASE_NAME}.gpt_only.log"
    else:
        if args.human_answers:
            votes = load_human_answers(args.human_answers)
            report += correlation_report(entries, items_by_number, votes)
            report += rater_agreement_report(entries, items_by_number, votes)
            report += strong_disagreement_report(entries, items_by_number, votes)
        elif args.calc_correlation_statistics_only:
            # The whole point of this flag is the statistics, and they need
            # votes. Saying so beats writing a report with the section missing.
            raise SystemExit(
                "--calc-correlation-statistics-only needs --human-answers.\n"
                "Build the CSV first:\n"
                "  python prepare_human_answers_csv.py\n"
                "then pass --human-answers human_answers.csv"
            )
        else:
            report += ("\nNo --human-answers given, so no correlation was "
                       "computed.\nBuild the CSV with "
                       "prepare_human_answers_csv.py, then re-run with\n"
                       "--calc-correlation-statistics-only --human-answers "
                       "human_answers.csv;\nit costs nothing.\n")
        log_path = args.out_dir / f"{BASE_NAME}.log"

    log_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"log -> {log_path}")


if __name__ == "__main__":
    main()

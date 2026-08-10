"""
STEP 6 -- LLM-as-a-judge (Prometheus) against the ORIGINAL SOURCE DOCUMENTS.

Unlike step 5 (which compares the candidate summaries against the reference
summaries provided by the datasets), this script judges the candidates against
the source texts/documents themselves, which is a better proxy for
faithfulness.

The judging prompt asks about faithfulness ONLY -- coverage, conciseness and
coherence are deliberately left out, so that the verdict is not diluted by
general summary quality. The prompt differs per dataset: cnn and xsum (news)
are judged on faithfulness alone, while wikihow (procedural how-to texts) gets
two extra criteria -- one for the specifics a step cannot be executed without
(quantities, durations, temperatures, tool names, settings), and one for step
order and prerequisite structure.

It automatically runs all 8 (checkpoints: 1M .. 8M) x 3 (datasets: cnn, xsum,
wikihow) = 24 PMI-vs-ROUGE comparisons, reading the candidate summaries
directly from the "eval_generated_pred/eval_results_*" folders.

Results are written as an independent JSON file per comparison (NOT merged into
the step4/step5 combined result files), plus a ".log" file holding the
aggregated result of that comparison. Per-dataset summary logs are written as
well.

The job is resumable: every judged sample is appended to a ".partial.jsonl"
file, so an interrupted run continues exactly where it stopped when restarted.
"""

import argparse
import contextlib
import json
import os
import random
from collections import Counter
from pathlib import Path

# cuBLAS chooses its reduction strategy through a workspace heuristic, which can
# differ between runs and change the low bits of every matmul. Pinning the
# workspace is only honoured when it is set BEFORE torch initialises CUDA, hence
# before the import below. enforce_determinism() holds the rest of the settings.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import transformers
from datasets import load_from_disk
from tqdm import tqdm
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from fastchat.conversation import get_conv_template

###############################################################################
# CONFIGURATION
###############################################################################

MODEL_NAME = "prometheus-eval/prometheus-7b-v2.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

MAX_NEW_TOKENS = 768

# Decoding is greedy (do_sample=False), so there is no temperature to lower: the
# verdict is an argmax over the logits, not a sample from them. What does move a
# verdict between runs is floating point -- on a near-tied token a difference in
# the last bits picks the other token, and greedy decoding then carries that
# divergence through the rest of the feedback. enforce_determinism() removes the
# sources of that jitter which are under this script's control; the batch size
# is the one that is not (see BATCH_SIZE).

# Which kernel torch.scaled_dot_product_attention runs. Left unpinned, it is
# picked per call by a heuristic that reads the shapes and the hardware, so the
# same sample can be computed by two different kernels -- and two kernels do not
# agree on the last bits. EFFICIENT_ATTENTION rather than FLASH_ATTENTION
# because the prompts are left-padded: flash takes no arbitrary attention mask
# and would be rejected, and pinning to a rejected backend raises instead of
# falling back. MATH is not an option either -- it builds the same seq x seq
# matrix that made eager OOM.
ATTENTION_BACKEND = SDPBackend.EFFICIENT_ATTENTION

# The bf16 weights alone are ~14.5 GB, which does not leave room for the KV
# cache on a 16 GB card (device_map="auto" would silently offload layers to the
# CPU and crawl). 4-bit NF4 brings the weights down to ~4.2 GB.
LOAD_IN_4BIT = True

# Source documents are much longer than the reference summaries used in step 5,
# so they are truncated before being put into the judging prompt. At 4000, only
# 24 of the 28401 test documents across the three datasets are truncated at all
# (cnn: 0, xsum: 7, wikihow: 17).
MAX_SOURCE_TOKENS = 4000

# Full source documents would blow up the output JSON files (hundreds of MB),
# so they are not stored by default. Flip to True if they are needed.
SAVE_SOURCE_DOCUMENT_IN_OUTPUT = False

# Set to an int to judge only the first N test samples of every comparison
# (useful for quick runs). None => judge the whole test set.
MAX_SAMPLES_PER_COMPARISON = None

# Samples per generate() call; override per machine with --batch-size.
#
# This is a numerical setting as much as a throughput one. With a batch larger
# than 1 a sample is padded up to the longest prompt it shares a batch with, so
# the matmul shapes -- and with them the low bits of its logits -- depend on its
# neighbours. Determinism therefore holds AT A FIXED BATCH SIZE: keep it
# constant across every comparison you intend to put side by side, and rerun a
# comparison rather than continuing it at a different size.
#
# Resuming re-cuts the batches on a grid starting at 0 (see
# run_single_comparison), so an interrupted run groups its samples exactly like
# an uninterrupted one. Every judged sample records the batch size it was
# produced with, which makes an accidentally mixed comparison visible.
BATCH_SIZE = 2

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

FINETUNE_DATA_DIR = REPO_ROOT / "finetune_data"
GENERATED_PRED_DIR = REPO_ROOT / "eval_generated_pred"

# Checkpoints (pretraining steps) to compare.
CHECKPOINTS = [f"{i}M" for i in range(1, 9)]

# dataset key -> where its source documents / candidate summaries / outputs live
#
# "prompt_style" selects the judging prompt (see build_judge_prompt): cnn and
# xsum are news articles and are judged on faithfulness alone, while wikihow is
# procedural and additionally gets a step-order / dependency criterion.
DATASETS = {
    "cnn": {
        "finetune_data_folder": "cnn_dailymail_comb",   # holds the source docs
        "eval_folder_suffix": "cnn_comb",               # eval_results_*_ft_<suffix>
        "result_folder": "cnn_result_files",            # where outputs are saved
        "prompt_style": "news",
    },
    "xsum": {
        "finetune_data_folder": "xsum_comb",
        "eval_folder_suffix": "xsum_comb",
        "result_folder": "xsum_result_files",
        "prompt_style": "news",
    },
    "wikihow": {
        "finetune_data_folder": "wikihow_comb",
        "eval_folder_suffix": "wikihow_comb",
        "result_folder": "wikihow_result_files",
        "prompt_style": "procedural",
    },
}

###############################################################################
# LOAD MODEL (lazily -- a fully resumed/finished run does not need the GPU)
###############################################################################

tokenizer = None
model = None
generation_config = None


def enforce_determinism():
    """
    Removes the run-to-run floating point jitter that this script can control.

    Greedy decoding makes every verdict an argmax, so a difference in the last
    bits of a logit is enough to flip a near-tied token and, through the rest of
    the greedy chain, the whole verdict. The settings below pin the parts of the
    numerics that would otherwise be chosen by a heuristic at runtime.

    What this does NOT make identical: a different GPU architecture, a different
    torch / transformers / bitsandbytes build, or a different batch size. 4-bit
    NF4 leaves little numerical headroom, so those do shift a few verdicts.
    environment_fingerprint() is written into every log to make such a change
    visible instead of silent.
    """

    # bitsandbytes has no deterministic implementation registered for some of
    # its kernels, so a hard failure would make the script unusable in 4-bit;
    # warn_only keeps the deterministic paths that do exist.
    torch.use_deterministic_algorithms(True, warn_only=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # else the algorithm depends on timings

    # TF32 and reduced-precision reductions are themselves reproducible, but
    # they throw away mantissa bits and so produce far more near-ties for the
    # argmax to flip on.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Nothing in the judging path samples, but a seed costs nothing and covers
    # any library that reaches for the global RNG during loading.
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


def attention_backend_context():
    """
    Restricts scaled_dot_product_attention to ATTENTION_BACKEND.

    Only on CUDA: the fused backends are CUDA kernels, so pinning one on a CPU
    run leaves the dispatcher with nothing viable and raises. A CPU run has only
    the math kernel anyway, which makes the choice deterministic by itself.
    """

    if not torch.cuda.is_available():
        return contextlib.nullcontext()

    return sdpa_kernel(ATTENTION_BACKEND)


def environment_fingerprint() -> str:
    """
    The versions and the device the numerics depend on, recorded in every log so
    that a shift in the verdicts can be traced to an environment change rather
    than mistaken for a difference between the checkpoints.
    """

    try:
        import bitsandbytes

        bnb_version = bitsandbytes.__version__
    except Exception:
        bnb_version = "n/a"

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        attention = ATTENTION_BACKEND.name.lower()
    else:
        device_name = "cpu"
        attention = "math (cpu)"

    return (
        f"torch={torch.__version__}, transformers={transformers.__version__}, "
        f"bitsandbytes={bnb_version}, device={device_name}, attn={attention}"
    )


def dtype_kwarg_name() -> str:
    """
    transformers renamed `torch_dtype` to `dtype` in 4.56 and dropped the old
    name in v5, so the correct keyword depends on the install. Picking it at
    runtime keeps this script working on machines with different versions.
    """

    try:
        major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    except ValueError:
        return "dtype"  # dev/rc version string -> assume something recent

    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def load_model_if_needed():
    global tokenizer, model, generation_config

    if model is not None:
        return

    print("Loading Prometheus (HF) with Mistral conversation template...")
    print(f"  -> {environment_fingerprint()}")

    enforce_determinism()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Needed for batching: decoder-only generation requires left padding, and
    # the Mistral tokenizer ships without a pad token.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map="auto" places the layers by looking at the FREE memory of the
    # moment, so a run with something else on the GPU can silently offload a
    # layer to the CPU and compute it in another precision. Pinning everything
    # to the first visible device keeps the placement identical between runs
    # (and still honours CUDA_VISIBLE_DEVICES).
    device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}

    # Pinned so the attention path is not whatever the install happens to
    # prefer. It must be sdpa, not eager: eager materialises the full
    # batch x heads x seq x seq score matrix, which at a 4000-token source
    # document and batch 2 is ~2.4 GB per layer in bf16 and OOMs a 16 GB card.
    # sdpa never builds that matrix; ATTENTION_BACKEND pins the kernel it uses.
    load_kwargs = {
        dtype_kwarg_name(): DTYPE,
        "device_map": device_map,
        "attn_implementation": "sdpa",
    }
    if LOAD_IN_4BIT and torch.cuda.is_available():
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_use_double_quant=True,
        )
        print("  -> loading in 4-bit NF4")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    model.eval()

    # Built explicitly rather than left to the checkpoint's
    # generation_config.json, which ships sampling defaults (temperature, top_p,
    # top_k) that would otherwise apply the moment do_sample were ever true.
    # Greedy + a single beam is the entire decoding policy of this script.
    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    print("Model loaded successfully.\n")

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
# PROMETHEUS JUDGE (versus the SOURCE DOCUMENT)
###############################################################################

def truncate_source_document(source_document: str) -> str:
    """Keeps the first MAX_SOURCE_TOKENS tokens of the source document."""

    token_ids = tokenizer(source_document, add_special_tokens=False)["input_ids"]

    if len(token_ids) <= MAX_SOURCE_TOKENS:
        return source_document

    return tokenizer.decode(token_ids[:MAX_SOURCE_TOKENS], skip_special_tokens=True)


# The judging prompt is faithfulness-only: coverage, conciseness and coherence
# were dropped because they pull the verdict towards general summary quality,
# which the other metrics of this project already measure, and because a judge
# weighing four criteria at once gives no interpretable signal about which one
# actually decided the comparison. They are simply left unmentioned rather than
# explicitly ruled out, so that the prompt never puts them in the judge's head.
# The one omission the wikihow prompt does look at is not coverage in that sense:
# it asks whether the steps a summary DOES mention keep the values they cannot be
# executed without, and is explicitly scoped to those steps only.
#
# The criteria are kept short on purpose: the judges are small models, and long
# enumerations of edge cases cost more attention than the detail is worth.
#
# Per prompt style: the system message, the criteria block, and the sentence
# naming the criteria in the task description.
PROMPT_STYLES = {
    # cnn / xsum -- news articles, single criterion.
    "news": {
        "system_message": (
            "You are a fair and precise evaluation assistant. "
            "You judge how faithful two candidate summaries are to the news article they "
            "were written from. "
            "Follow the evaluation criterion carefully and be impartial."
        ),
        "criteria_word": "criterion",
        "criteria_header": "EVALUATION CRITERION (this is the ONLY thing you judge):",
        "criteria_block": (
            "**Faithfulness:** Is every statement in the summary supported by the Source Document?\n"
            "Names, numbers, dates, places, quotes and outcomes must all be traceable to the Source\n"
            "Document. Count as unfaithful anything the summary invents, and anything that\n"
            "contradicts the Source Document."
        ),
    },
    # wikihow -- procedural how-to texts, so the exact values of a step and the
    # order and prerequisite structure of the steps carry meaning that a news
    # summary does not have: a dropped temperature or a swapped step makes the
    # procedure unusable even when nothing in the summary is actually false.
    "procedural": {
        "system_message": (
            "You are a fair and precise evaluation assistant. "
            "You judge how faithful two candidate summaries are to the how-to article they "
            "were written from, including whether they get the procedure right. "
            "Follow the evaluation criteria carefully and be impartial."
        ),
        "criteria_word": "criteria",
        "criteria_header": "EVALUATION CRITERIA (these are the ONLY things you judge):",
        "criteria_block": (
            "1. **Faithfulness:** Is every statement in the summary supported by the Source Document, without hallucinated or contradicting information?\n"
            "\n"
            "2. **Critical Specifics:** For every step the summary does mention, does it keep the\n"
            "details that step cannot be carried out without: quantities, durations, temperatures,\n"
            "sizes, settings, and the specific tool, material or ingredient names? Dropping one such\n"
            "detail, or replacing it with a vague phrase (\"bake it\" instead of \"bake at 180 C for 20\n"
            "minutes\"), is a serious flaw, because the reader can no longer perform the step.\n"
            "\n"
            "3. **Procedural Order:** Do the steps appear in the same order as in the Source\n"
            "Document, and does every step still come after whatever it depends on?\n"
        ),
    },
}

"""
"\n"
            "Weigh criterion 1 first, then criterion 2, then criterion 3. Judge criterion 2 only on\n"
            "the steps a summary actually mentions."
"""


def build_judge_prompt(
    source_document: str,
    pmi_summary: str,
    rouge_summary: str,
    swap: bool,
    dataset_key: str,
) -> str:
    """
    Builds the full judging prompt for one sample.

    `swap` decides the position of each candidate, so that positional bias is
    averaged out. It is derived deterministically by the caller, which keeps a
    resumed run identical to an uninterrupted one.

    `dataset_key` selects the prompt style (see PROMPT_STYLES): news datasets
    are judged on faithfulness alone, wikihow also on procedural order.
    """

    if swap:
        response_A = pmi_summary
        response_B = rouge_summary
    else:
        response_A = rouge_summary
        response_B = pmi_summary

    source_document = truncate_source_document(source_document)

    style = PROMPT_STYLES[DATASETS[dataset_key]["prompt_style"]]

    conv = get_conv_template("mistral")
    conv.set_system_message(style["system_message"])

    instruction = f"""
TASK DESCRIPTION:
1. You are given a Source Document and two Candidate Summaries (A and B) of that document.
2. Decide which Candidate Summary is better according to the evaluation {style["criteria_word"]} below, and nothing else.
3. Compare both summaries statement by statement against the Source Document, and base your decision on the violations of the {style["criteria_word"]} that you can actually point to in the text.
4. Write a brief feedback that assess the two candidate summaries strictly based on the given evaluation {style["criteria_word"]}, not evaluating in general.
5. Prefer a decision over a "TIE": if one summary is even slightly better on the evaluation {style["criteria_word"]} - fewer violations, or a violation that is less damaging to the reader - choose that summary. Answer "TIE" only as a last resort, when you cannot point to any difference at all between the two summaries with respect to the {style["criteria_word"]}.
6. After writing the feedback, indicate the better candidate summary, either "A" or "B" or "TIE".
7. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B" or "TIE")"
8. Please do not generate any other opening, closing, and explanations.

{style["criteria_header"]}
{style["criteria_block"]}

SOURCE DOCUMENT:
{source_document}

CANDIDATE A:
{response_A}

CANDIDATE B:
{response_B}

FEEDBACK:
""".strip()

    conv.append_message(conv.roles[0], instruction)

    return conv.get_prompt()


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


def judge_batch_with_prometheus(batch, dataset_key: str):
    """
    Judges a batch of samples in one generate() call.

    `batch` is a list of (source_document, pmi_summary, rouge_summary, swap)
    tuples. Returns a list of (winner, feedback, raw_result) in the same order.
    `dataset_key` picks the judging prompt, so a batch must not mix datasets.

    A batch size of 1 reproduces the original one-at-a-time behaviour. Larger
    batches keep the GPU busy during decoding, which is where nearly all of the
    time goes.
    """

    prompts = [
        build_judge_prompt(source_document, pmi_summary, rouge_summary, swap, dataset_key)
        for source_document, pmi_summary, rouge_summary, swap in batch
    ]

    # Left padding: decoder-only models must have the prompts end flush against
    # the generated tokens, otherwise the continuations start after the padding.
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(model.device)

    # sdpa_kernel restricts the dispatcher to the one backend for the whole
    # generation, prefill and decoding alike (see ATTENTION_BACKEND).
    with torch.no_grad(), attention_backend_context():
        outputs = model.generate(**inputs, generation_config=generation_config)

    # Decode ONLY the newly generated tokens. Decoding the whole sequence would
    # drag the prompt (and with it the entire source document) into `feedback`,
    # and would break [RESULT] detection, because the prompt itself contains the
    # literal "[RESULT]" in its output-format instructions.
    completions = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    results = []
    for completion, (_, _, _, swap) in zip(completions, batch):
        feedback, raw_result = parse_prometheus_output(completion)
        results.append((decode_winner(raw_result, swap), feedback, raw_result))

    return results

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
            if entry.get("index") != len(entries):
                break  # gap -> stop here and re-judge from this point on
            entries.append(entry)

    return entries


def rewrite_partial_file(partial_path: Path, entries):
    """Rewrites the partial file so it exactly matches the kept entries."""

    with open(partial_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def aggregate_text(dataset_key: str, checkpoint: str, entries, batch_size: int = BATCH_SIZE) -> str:
    counts = Counter(entry["llm_judge_winner"] for entry in entries)
    total = len(entries)

    # Generations that never produced a [RESULT] tag. They fall into "tie", so
    # a high rate here means MAX_NEW_TOKENS is cutting the judge off.
    unparsed = sum(1 for entry in entries if entry.get("raw_result") == "TIE_2")

    # Reported from the entries themselves: a comparison that was resumed at a
    # different batch size shows both sizes here (see BATCH_SIZE).
    used_sizes = sorted(
        {entry.get("batch_size", batch_size) for entry in entries},
        key=lambda size: (size is None, size),
    )

    lines = [
        f"DATASET   : {dataset_key}",
        f"CHECKPOINT: {checkpoint}",
        f"JUDGE     : {MODEL_NAME} (compared against the SOURCE DOCUMENTS)",
        f"PROMPT    : {DATASETS[dataset_key]['prompt_style']}",
        f"SETTINGS  : max_source_tokens={MAX_SOURCE_TOKENS}, max_new_tokens={MAX_NEW_TOKENS}, "
        f"4bit={LOAD_IN_4BIT}, greedy=True, "
        f"batch_size={','.join(str(size) for size in used_sizes)}",
        f"ENV       : {environment_fingerprint()}",
        f"SAMPLES   : {total}",
        "----------------------",
        f"PMI wins   : {counts['pmi']}  ({counts['pmi'] / total * 100:.4f}%)",
        f"ROUGE wins : {counts['rouge']} ({counts['rouge'] / total * 100:.4f}%)",
        f"TIES       : {counts['tie']} ({counts['tie'] / total * 100:.4f}%)",
        "----------------------",
        f"  of which no [RESULT] tag : {unparsed} ({unparsed / total * 100:.4f}%)",
    ]
    return "\n".join(lines)

def write_dataset_summary(dataset_key: str):
    """
    (Re)builds the per-dataset summary log from whatever comparison outputs are
    actually on disk.

    Rebuilding from disk instead of from the current run means the summary stays
    complete when the 8 checkpoints are split across several runs or devices --
    a run that only does 5M..8M will not drop the 1M..4M lines written earlier.
    """

    result_dir = SCRIPT_DIR / DATASETS[dataset_key]["result_folder"]

    lines = [
        f"LLM-as-a-judge ({MODEL_NAME}) versus SOURCE DOCUMENTS -- step 6",
        f"DATASET: {dataset_key}",
        f"PROMPT : {DATASETS[dataset_key]['prompt_style']}",
        # The summary is rebuilt from the JSON outputs, which carry no
        # environment of their own, so this is the machine doing the rebuild.
        # The env each checkpoint was actually judged on is in its own .log.
        f"ENV    : {environment_fingerprint()} (rebuild host)",
        "=" * 70,
    ]

    for checkpoint in CHECKPOINTS:
        path = result_dir / f"{dataset_key}_{checkpoint}_llm_judge_vs_source_docs__step6.json"

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
        total = len(entries)

        lines.append(
            f"{checkpoint}: samples={total} | "
            f"pmi={counts['pmi']} ({counts['pmi'] / total * 100:.4f}%) | "
            f"rouge={counts['rouge']} ({counts['rouge'] / total * 100:.4f}%) | "
            f"tie={counts['tie']} ({counts['tie'] / total * 100:.4f}%) | "
            f"no_result_tag={unparsed} ({unparsed / total * 100:.4f}%)"
        )

    summary_path = (
        result_dir
        / f"{dataset_key}_ALL_checkpoints_llm_judge_vs_source_docs__step6_summary.log"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

###############################################################################
# COMMAND LINE SELECTION (for splitting the job across devices)
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
            "Step 6 -- Prometheus LLM-as-a-judge of PMI- vs ROUGE-pegasus summaries "
            "against the original source documents. Without arguments it runs all "
            "3 datasets x 8 checkpoints. Use the flags to split the job across "
            "devices, e.g. --checkpoints 1M,2M,3M,4M on one machine and "
            "--checkpoints 5M,6M,7M,8M on another."
        )
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help=f"comma separated subset of: {', '.join(DATASETS)} (default: all)",
    )
    parser.add_argument(
        "--checkpoints",
        default="all",
        help=f"comma separated subset of: {', '.join(CHECKPOINTS)} (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(
            f"samples judged per generate() call (default: {BATCH_SIZE}). "
            "Padding makes this part of the numerics, so verdicts are only "
            "reproducible at a fixed batch size: keep it constant across every "
            "comparison you intend to put side by side, and rerun rather than "
            "resume a comparison if you change it."
        ),
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")

    return args

###############################################################################
# SINGLE COMPARISON (one dataset + one checkpoint)
###############################################################################

def run_single_comparison(dataset_key: str, checkpoint: str, batch_size: int = BATCH_SIZE):
    config = DATASETS[dataset_key]

    result_dir = SCRIPT_DIR / config["result_folder"]
    result_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{dataset_key}_{checkpoint}_llm_judge_vs_source_docs__step6"
    output_path = result_dir / f"{base_name}.json"
    partial_path = result_dir / f"{base_name}.partial.jsonl"
    log_path = result_dir / f"{base_name}.log"

    test_set_path = FINETUNE_DATA_DIR / config["finetune_data_folder"] / "test"
    pmi_path = (
        GENERATED_PRED_DIR
        / f"eval_results_PMI_pegasus_complete_{checkpoint}_pt_100k_ft_{config['eval_folder_suffix']}"
        / "generated_predictions.txt"
    )
    rouge_path = (
        GENERATED_PRED_DIR
        / f"eval_results_ROUGE_pegasus_complete_{checkpoint}_pt_100k_ft_{config['eval_folder_suffix']}"
        / "generated_predictions.txt"
    )

    for path in (test_set_path, pmi_path, rouge_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    ds = load_from_disk(str(test_set_path))
    source_documents = ds["document"]
    doc_ids = ds["id"] if "id" in ds.column_names else list(range(len(ds)))

    with open(pmi_path, "r", encoding="utf-8") as f:
        pmi_summaries = [line.strip() for line in f]

    with open(rouge_path, "r", encoding="utf-8") as f:
        rouge_summaries = [line.strip() for line in f]

    if not (len(source_documents) == len(pmi_summaries) == len(rouge_summaries)):
        raise ValueError(
            f"Size mismatch for {dataset_key} {checkpoint}: "
            f"source docs={len(source_documents)}, pmi={len(pmi_summaries)}, "
            f"rouge={len(rouge_summaries)}"
        )

    total_samples = len(source_documents)
    if MAX_SAMPLES_PER_COMPARISON is not None:
        total_samples = min(total_samples, MAX_SAMPLES_PER_COMPARISON)

    # ---- already finished? -------------------------------------------------
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            finished_entries = json.load(f)
        if len(finished_entries) == total_samples:
            print(f"[SKIP] {dataset_key} {checkpoint}: already complete "
                  f"({total_samples} samples) -> {output_path.name}")
            return finished_entries
        print(f"[REDO] {dataset_key} {checkpoint}: existing output has "
              f"{len(finished_entries)} / {total_samples} samples, continuing...")

    # ---- resume from the partial file --------------------------------------
    entries = read_partial_results(partial_path)
    entries = entries[:total_samples]

    # Batches are always cut on a grid starting at 0, so a resumed run pads its
    # samples exactly like an uninterrupted one (see BATCH_SIZE). A partial file
    # that stops off-grid -- because the previous run used a different batch
    # size -- is rolled back to the last boundary, and those few samples are
    # judged again under the current size.
    if len(entries) < total_samples and len(entries) % batch_size:
        aligned = len(entries) - (len(entries) % batch_size)
        print(f"[ALIGN] {dataset_key} {checkpoint}: dropping "
              f"{len(entries) - aligned} judged sample(s) so the batches "
              f"restart on a multiple of {batch_size}")
        entries = entries[:aligned]

    foreign_sizes = {entry.get("batch_size") for entry in entries} - {batch_size, None}
    if foreign_sizes:
        print(f"[WARN] {dataset_key} {checkpoint}: this comparison already "
              f"contains samples judged at batch size(s) "
              f"{sorted(foreign_sizes)}, now continuing at {batch_size}. The "
              f"two halves are not numerically comparable -- delete "
              f"{partial_path.name} to re-judge the comparison in one go.")

    rewrite_partial_file(partial_path, entries)

    start_index = len(entries)
    if start_index > 0:
        print(f"[RESUME] {dataset_key} {checkpoint}: continuing at sample "
              f"{start_index} / {total_samples}")

    if start_index < total_samples:
        load_model_if_needed()

    partial_file = open(partial_path, "a", encoding="utf-8")
    try:
        progress = tqdm(
            total=total_samples,
            initial=start_index,
            desc=f"{dataset_key} {checkpoint} (bs={batch_size})",
        )

        for batch_start in range(start_index, total_samples, batch_size):
            batch_indices = list(
                range(batch_start, min(batch_start + batch_size, total_samples))
            )

            # Deterministic per-sample position swap -> a resumed run makes the
            # exact same A/B assignment as an uninterrupted one, and the batch
            # size does not change which candidate sits in which position.
            swaps = [
                random.Random(f"{dataset_key}|{checkpoint}|{i}").random() < 0.5
                for i in batch_indices
            ]

            batch = [
                (source_documents[i], pmi_summaries[i], rouge_summaries[i], swap)
                for i, swap in zip(batch_indices, swaps)
            ]

            results = judge_batch_with_prometheus(batch, dataset_key)

            for i, swap, (winner, feedback, raw_result) in zip(
                batch_indices, swaps, results
            ):
                entry = {
                    "index": i,
                    "id": doc_ids[i],
                    "pmi_summary": pmi_summaries[i],
                    "rouge_summary": rouge_summaries[i],
                    "pmi_position": "A" if swap else "B",
                    "llm_judge_winner": winner,
                    "raw_result": raw_result,
                    "llm_judge_feedback": feedback,
                    # Padding makes the batch size part of the numerics, so it
                    # is recorded per sample rather than per run.
                    "batch_size": batch_size,
                }
                if SAVE_SOURCE_DOCUMENT_IN_OUTPUT:
                    entry["source_document"] = source_documents[i]

                entries.append(entry)
                partial_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Flushed once per batch: an interrupted run loses at most the
            # current batch, which is then simply re-judged on resume.
            partial_file.flush()
            os.fsync(partial_file.fileno())

            progress.update(len(batch_indices))

        progress.close()
    finally:
        partial_file.close()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)

    aggregate = aggregate_text(dataset_key, checkpoint, entries, batch_size)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(aggregate + "\n")

    print("\n" + aggregate + "\n")

    # The partial file is only a crash-safety net -> drop it once the final
    # JSON output is on disk.
    partial_path.unlink(missing_ok=True)

    return entries

###############################################################################
# MAIN ENTRY POINT (CNN + XSUM + WIKIHOW, checkpoints 1M .. 8M)
###############################################################################

if __name__ == "__main__":

    args = parse_args()
    selected_datasets = select_datasets(args.datasets)
    selected_checkpoints = select_checkpoints(args.checkpoints)

    print(f"Datasets   : {', '.join(selected_datasets)}")
    print(f"Checkpoints: {', '.join(selected_checkpoints)}")
    print(f"Batch size : {args.batch_size}")
    print(f"=> {len(selected_datasets) * len(selected_checkpoints)} comparison(s)")

    overall_summary = {}

    for dataset_key in selected_datasets:
        for checkpoint in selected_checkpoints:
            print(f"\n\n{'*' * 70}")
            print(f"***  {dataset_key.upper()}  --  {checkpoint} pretraining steps")
            print(f"{'*' * 70}\n")

            entries = run_single_comparison(dataset_key, checkpoint, args.batch_size)

            overall_summary[(dataset_key, checkpoint)] = Counter(
                entry["llm_judge_winner"] for entry in entries
            )

            # Rewritten after every checkpoint so partial progress is visible.
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

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
an extra criterion for step order and prerequisite structure.

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
import json
import os
import random
from collections import Counter
from pathlib import Path

import torch
import transformers
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastchat.conversation import get_conv_template

###############################################################################
# CONFIGURATION
###############################################################################

MODEL_NAME = "prometheus-eval/prometheus-7b-v2.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

MAX_NEW_TOKENS = 768
TEMPERATURE = 0.01  # deterministic judging

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

# Samples per generate() call. 1 keeps the original one-at-a-time behaviour;
# override per machine with --batch-size.
#
# Measured on an RTX 5080 (16 GB, 4-bit NF4, wikihow, 24 samples):
#   bs=1  -> 4.21 s/sample      bs=8  -> 4.49 s/sample (0.94x, i.e. no gain)
#   bs=16 -> CUDA OOM
# Batching does not pay off here: the prompts are long, so prefill work grows
# linearly with the batch, and a batch keeps generating until its LONGEST
# completion is done, which wastes the slots that finished early. Batch sizes
# above 1 also shift a few verdicts (see below), so 1 is the default.
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
    global tokenizer, model

    if model is not None:
        return

    print("Loading Prometheus (HF) with Mistral conversation template...")
    print(f"  -> transformers {transformers.__version__}, torch {torch.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Needed for batching: decoder-only generation requires left padding, and
    # the Mistral tokenizer ships without a pad token.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {dtype_kwarg_name(): DTYPE, "device_map": "auto"}
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
    # wikihow -- procedural how-to texts, so the order and the prerequisite
    # structure of the steps carry meaning that a news summary does not have.
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
            "1. **Faithfulness:** Is every statement in the summary supported by the Source Document?\n"
            "Actions, materials, tools, quantities and warnings must all be traceable to the Source\n"
            "Document. Count as unfaithful anything the summary invents, and anything that\n"
            "contradicts the Source Document.\n"
            "\n"
            "2. **Procedural Order:** Do the steps appear in the same order as in the Source\n"
            "Document, and does every step still come after whatever it depends on?\n"
            "\n"
            "Weigh criterion 1 first. Use criterion 2 to decide when both summaries are comparably\n"
            "faithful."
        ),
    },
}


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
5. If both summaries violate the evaluation {style["criteria_word"]} to the same degree, or if neither violates them at all, answer "TIE".
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

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

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

    lines = [
        f"DATASET   : {dataset_key}",
        f"CHECKPOINT: {checkpoint}",
        f"JUDGE     : {MODEL_NAME} (compared against the SOURCE DOCUMENTS)",
        f"PROMPT    : {DATASETS[dataset_key]['prompt_style']}",
        f"SETTINGS  : max_source_tokens={MAX_SOURCE_TOKENS}, max_new_tokens={MAX_NEW_TOKENS}, "
        f"4bit={LOAD_IN_4BIT}, batch_size={batch_size}",
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
            "samples judged per generate() call (default: 1). Measured on an "
            "RTX 5080 this gave no speedup (bs=8 was 0.94x) and flipped 4 of 24 "
            "verdicts versus bs=1, because padding changes the numerics. Only "
            "raise it if you have benchmarked a gain on your own GPU, and then "
            "keep it fixed for every comparison you intend to compare."
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

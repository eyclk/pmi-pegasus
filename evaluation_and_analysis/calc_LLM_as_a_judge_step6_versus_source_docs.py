"""
STEP 6 -- LLM-as-a-judge (Prometheus) against the ORIGINAL SOURCE DOCUMENTS.

Unlike step 5 (which compares the candidate summaries against the reference
summaries provided by the datasets), this script judges the candidates against
the source texts/documents themselves, which is a better proxy for
faithfulness.

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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

FINETUNE_DATA_DIR = REPO_ROOT / "finetune_data"
GENERATED_PRED_DIR = REPO_ROOT / "eval_generated_pred"

# Checkpoints (pretraining steps) to compare.
CHECKPOINTS = [f"{i}M" for i in range(1, 9)]

# dataset key -> where its source documents / candidate summaries / outputs live
DATASETS = {
    "cnn": {
        "finetune_data_folder": "cnn_dailymail_comb",   # holds the source docs
        "eval_folder_suffix": "cnn_comb",               # eval_results_*_ft_<suffix>
        "result_folder": "cnn_result_files",            # where outputs are saved
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
# LOAD MODEL (lazily -- a fully resumed/finished run does not need the GPU)
###############################################################################

tokenizer = None
model = None


def load_model_if_needed():
    global tokenizer, model

    if model is not None:
        return

    print("Loading Prometheus (HF) with Mistral conversation template...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    load_kwargs = {"dtype": DTYPE, "device_map": "auto"}
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


def judge_with_prometheus(
    source_document: str,
    pmi_summary: str,
    rouge_summary: str,
    swap: bool,
) -> (str, str, str):
    """
    Returns (decoded winner: 'pmi' / 'rouge' / 'tie', feedback, raw_result).

    `swap` decides the position of each candidate, so that positional bias is
    averaged out. It is derived deterministically by the caller, which keeps a
    resumed run identical to an uninterrupted one.
    """

    if swap:
        response_A = pmi_summary
        response_B = rouge_summary
    else:
        response_A = rouge_summary
        response_B = pmi_summary

    source_document = truncate_source_document(source_document)

    conv = get_conv_template("mistral")
    conv.set_system_message(
        "You are a fair and precise evaluation assistant. "
        "You compare two candidate summaries against the source document they were written from. "
        "Follow the evaluation criteria carefully and be impartial."
    )

    instruction = f"""
TASK DESCRIPTION:
1. You are given a Source Document and two Candidate Summaries (A and B) of that document.
2. Your task is to evaluate the quality of the two Candidate Summaries based on the Source Document using the specified Evaluation Criteria.
3. Write a brief feedback that assess the quality of the two candidate summaries strictly based on the given evaluation criteria, not evaluating in general.
4. After writing the feedback, indicate the better candidate summary, either "A" or "B" or "TIE".
5. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B" or "TIE")"
6. Please do not generate any other opening, closing, and explanations.

EVALUATION CRITERIA:
1. **Faithfulness:** Is every statement in the summary supported by the Source Document, without hallucinated or contradicting information?
2. **Coverage:** How well does the summary capture the essential points of the Source Document?
3. **Conciseness:** Is the summary brief without sacrificing key details of the Source Document?
4. **Coherence:** Is the summary easy to read and logically organized?

SOURCE DOCUMENT:
{source_document}

CANDIDATE A:
{response_A}

CANDIDATE B:
{response_B}

FEEDBACK:
""".strip()

    conv.append_message(conv.roles[0], instruction)

    prompt = conv.get_prompt()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    feedback, raw_result = parse_prometheus_output(decoded_output)

    # Decode winner back to PMI / ROUGE
    if swap:
        decoded_winner = (
            "pmi" if raw_result == "A"
            else "rouge" if raw_result == "B"
            else "tie"
        )
    else:
        decoded_winner = (
            "rouge" if raw_result == "A"
            else "pmi" if raw_result == "B"
            else "tie"
        )

    return decoded_winner, feedback, raw_result

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


def aggregate_text(dataset_key: str, checkpoint: str, entries) -> str:
    counts = Counter(entry["llm_judge_winner"] for entry in entries)
    total = len(entries)

    # Generations that never produced a [RESULT] tag. They fall into "tie", so
    # a high rate here means MAX_NEW_TOKENS is cutting the judge off.
    unparsed = sum(1 for entry in entries if entry.get("raw_result") == "TIE_2")

    lines = [
        f"DATASET   : {dataset_key}",
        f"CHECKPOINT: {checkpoint}",
        f"JUDGE     : {MODEL_NAME} (compared against the SOURCE DOCUMENTS)",
        f"SETTINGS  : max_source_tokens={MAX_SOURCE_TOKENS}, max_new_tokens={MAX_NEW_TOKENS}, 4bit={LOAD_IN_4BIT}",
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
    return parser.parse_args()

###############################################################################
# SINGLE COMPARISON (one dataset + one checkpoint)
###############################################################################

def run_single_comparison(dataset_key: str, checkpoint: str):
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
            range(start_index, total_samples),
            total=total_samples,
            initial=start_index,
            desc=f"{dataset_key} {checkpoint}",
        )

        for i in progress:
            # Deterministic per-sample position swap -> a resumed run makes the
            # exact same A/B assignment as an uninterrupted one.
            swap = random.Random(f"{dataset_key}|{checkpoint}|{i}").random() < 0.5

            winner, feedback, raw_result = judge_with_prometheus(
                source_documents[i],
                pmi_summaries[i],
                rouge_summaries[i],
                swap,
            )

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
            partial_file.flush()
            os.fsync(partial_file.fileno())
    finally:
        partial_file.close()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)

    aggregate = aggregate_text(dataset_key, checkpoint, entries)
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
    print(f"=> {len(selected_datasets) * len(selected_checkpoints)} comparison(s)")

    overall_summary = {}

    for dataset_key in selected_datasets:
        for checkpoint in selected_checkpoints:
            print(f"\n\n{'*' * 70}")
            print(f"***  {dataset_key.upper()}  --  {checkpoint} pretraining steps")
            print(f"{'*' * 70}\n")

            entries = run_single_comparison(dataset_key, checkpoint)

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

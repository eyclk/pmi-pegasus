import argparse
import math

import nltk
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer

SENTENCE_BATCH_SIZE = 64

USE_SMALLER_SUBSET = True  # MODIFY HERE TO USE A SMALLER SUBSET OF THE DATASET
SUBSET_LOWER_LIMIT = 0
SUBSET_UPPER_LIMIT = 1000000

# Option B baseline: fixed generic source prompt for "no document info"
BASELINE_SOURCE_PROMPT = "Document:"

# Use per-token normalization to reduce sentence-length bias in importance scoring
USE_PER_TOKEN_NORMALIZATION = True

# Keep targets modest; sentences rarely need 512 tokens
SOURCE_MAX_LENGTH = 512
TARGET_MAX_LENGTH = 128

parser = argparse.ArgumentParser()
parser.add_argument(
    "--c4_split", type=str, default="realnewslike", choices=["en", "realnewslike"]
)
parser.add_argument("--topk", type=int, default=5)
args = parser.parse_args()

OUTPUT_PATH = "./PREPROCESSED_DATASETS/c4_{}_processed_PMI_0_to_1_mil__FAST".format(
    args.c4_split
)

mask_token = "<mask>"

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Load pre-trained model (weights)
model = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-base", forced_bos_token_id=0
)

# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

all_PMI_scores_dict = {}


def seq_logscore_bart(
    source_texts,
    target_texts,
    *,
    tokenizer,
    model,
    device,
    source_max_length=512,
    target_max_length=128,
    per_token_normalization=True,
):
    """
    Compute a log-score for each (source, target) pair under BART using teacher forcing.

    Returns: tensor shape (batch_size,), where higher is better.
    - If per_token_normalization=True: returns average logprob per target token.
    - Else: returns total logprob of the target sequence.

    Note: This is a *score* derived from the model's token logprobs, not a calibrated probability.
    """
    src = tokenizer(
        source_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=source_max_length,
    ).to(device)

    # target encoding
    with tokenizer.as_target_tokenizer():
        tgt = tokenizer(
            target_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=target_max_length,
        ).to(device)

    labels = tgt["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        out = model(
            input_ids=src["input_ids"],
            attention_mask=src["attention_mask"],
            labels=labels,
            return_dict=True,
        )
        logits = out.logits  # (bs, tgt_len, vocab)

        nll_tok = F.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.shape)

        valid = (labels != -100).float()
        nll_sum = (nll_tok * valid).sum(dim=1)

        if per_token_normalization:
            tok_count = valid.sum(dim=1).clamp_min(1.0)
            nll = nll_sum / tok_count
        else:
            nll = nll_sum

        log_score = -nll  # higher is better

    return log_score


def calculate_importance_pmi(
    target_sentences_per_text,
    docs_without_target_sentences_per_text,
    *,
    tokenizer,
    model,
    device,
    baseline_source_prompt="Document:",
):
    """
    Option B baseline PMI for "importance of sentence to document":

    PMI_bits = log2 P(s | d \ s) - log2 P(s | baseline_prompt)

    Uses teacher-forced scoring only (no generation).
    """
    log_cond = seq_logscore_bart(
        docs_without_target_sentences_per_text,
        target_sentences_per_text,
        tokenizer=tokenizer,
        model=model,
        device=device,
        source_max_length=SOURCE_MAX_LENGTH,
        target_max_length=TARGET_MAX_LENGTH,
        per_token_normalization=USE_PER_TOKEN_NORMALIZATION,
    )

    baseline_sources = [baseline_source_prompt] * len(target_sentences_per_text)
    log_base = seq_logscore_bart(
        baseline_sources,
        target_sentences_per_text,
        tokenizer=tokenizer,
        model=model,
        device=device,
        source_max_length=SOURCE_MAX_LENGTH,
        target_max_length=TARGET_MAX_LENGTH,
        per_token_normalization=USE_PER_TOKEN_NORMALIZATION,
    )

    # Convert from natural log to log2
    inv_ln2 = 1.0 / math.log(2.0)
    pmi_bits = (log_cond - log_base) * inv_ln2
    return pmi_bits.detach().cpu().tolist()


def single_process_calc_pmi_for_all(training_dataset):
    all_sentences = []
    texts_corresponding_to_sentences = []

    for s in tqdm.tqdm(training_dataset, desc="Splitting all texts into separate sentences"):
        temp_text = s["text"]
        sentences = nltk.sent_tokenize(temp_text)
        all_sentences.extend(sentences)
        texts_corresponding_to_sentences.extend([temp_text] * len(sentences))

    # Batch all sentences according to SENTENCE_BATCH_SIZE and score them
    for i in tqdm.tqdm(range(0, len(all_sentences), SENTENCE_BATCH_SIZE), desc="Calculating PMI (no generation)"):
        sentences_per_batch = all_sentences[i : i + SENTENCE_BATCH_SIZE]
        texts_per_batch = texts_corresponding_to_sentences[i : i + SENTENCE_BATCH_SIZE]

        # remove the sentence once (same as your original logic)
        docs_per_batch = [t.replace(s, "", 1) for t, s in zip(texts_per_batch, sentences_per_batch)]

        pmi_scores_per_batch = calculate_importance_pmi(
            sentences_per_batch,
            docs_per_batch,
            tokenizer=tokenizer,
            model=model,
            device=device,
            baseline_source_prompt=BASELINE_SOURCE_PROMPT,
        )

        for score, text in zip(pmi_scores_per_batch, texts_per_batch):
            if all_PMI_scores_dict.get(text) is None:
                all_PMI_scores_dict[text] = [score]
            else:
                all_PMI_scores_dict[text].append(score)


def calc_pmi_score_and_select_top_k(example):
    temp_text = example["text"]
    sentences = nltk.sent_tokenize(temp_text)
    scores = all_PMI_scores_dict[temp_text]

    # top k
    if len(scores) <= args.topk:
        ind = np.arange(len(scores))
    else:
        ind = np.argpartition(scores, -args.topk)[-args.topk:]

    example["documents"] = [
        " ".join([s if j != i else mask_token for j, s in enumerate(sentences)])
        for i in ind
    ]
    example["summaries"] = [sentences[i] for i in ind]
    example["pmi"] = [scores[i] for i in ind]

    return example


if __name__ == "__main__":
    dataset = load_dataset("c4", args.c4_split, cache_dir="./cache")
    dataset.pop("validation")

    if USE_SMALLER_SUBSET:
        if SUBSET_UPPER_LIMIT > len(dataset["train"]):
            SUBSET_UPPER_LIMIT = len(dataset["train"])
        dataset["train"] = dataset["train"].select(list(range(SUBSET_LOWER_LIMIT, SUBSET_UPPER_LIMIT)))

    single_process_calc_pmi_for_all(dataset["train"])

    dataset["train"] = dataset["train"].map(
        calc_pmi_score_and_select_top_k,
        remove_columns=["url", "text", "timestamp"],
        batched=False,
        num_proc=16,
        keep_in_memory=True,
    )

    dataset.save_to_disk(OUTPUT_PATH)
"""
Computes the statistics of the three finetuning datasets (XSum, CNN/DailyMail, WikiHow) for the dataset table
of the paper, and prints them together with a ready-to-paste LaTeX table.

The splits are read directly from the Arrow files written by `datasets.save_to_disk`, so only pyarrow and nltk
are needed (the `datasets` library is not imported).

Conventions (chosen so that our XSum numbers reproduce the published XSum statistics of Narayan et al., 2018):
  - Sentences: each text is split on newlines first, then each line is split with NLTK's Punkt sentence tokenizer.
  - Words: NLTK Treebank tokens, punctuation included (the same tokens nltk.word_tokenize returns).
  - Novel bigrams: per example, the % of unique lowercased summary bigrams that do not occur in the document,
    averaged over all examples.
  - WikiHow paragraphs were concatenated without a space ("...and decorations.If the cage is..."), so a space is
    inserted between sentence-final punctuation and a following capital letter before tokenizing.
Split sizes are reported per split; every other statistic is computed over train + validation + test.
"""
import json
import re
from multiprocessing import Pool
from pathlib import Path

import nltk
import pyarrow as pa
from nltk.tokenize import NLTKWordTokenizer, sent_tokenize
from tqdm import tqdm

FINETUNE_DATA_DIR = Path(__file__).resolve().parents[2] / "finetune_data"
DATASETS = {
    "XSum": {
        "path": FINETUNE_DATA_DIR / "xsum_comb",
        "domain": "News",
    },
    "CNN/DailyMail": {
        "path": FINETUNE_DATA_DIR / "cnn_dailymail_comb",
        "domain": "News",
    },
    "WikiHow": {
        "path": FINETUNE_DATA_DIR / "wikihow_comb",
        "domain": "Procedural",
    },
}
SPLITS = ["train", "validation", "test"]
NUM_WORKERS = 12
CHUNK_SIZE = 256

MISSING_SPACE_AFTER_SENTENCE = re.compile(r"(?<=[a-z0-9][.!?])(?=[A-Z])")
word_tokenizer = NLTKWordTokenizer()


def iter_split(split_dir):
    """Yields (document, summary) pairs from a split saved with `datasets.save_to_disk`."""
    with open(split_dir / "state.json", encoding="utf-8") as f:
        data_files = [entry["filename"] for entry in json.load(f)["_data_files"]]
    for filename in data_files:
        with pa.memory_map(str(split_dir / filename)) as source:
            for batch in pa.ipc.open_stream(source):
                yield from zip(batch.column("document").to_pylist(), batch.column("summary").to_pylist())


def split_sentences(text):
    text = MISSING_SPACE_AFTER_SENTENCE.sub(" ", text)
    return [sent for line in text.split("\n") if line.strip() for sent in sent_tokenize(line)]


def tokenize(sentences):
    return [token for sent in sentences for token in word_tokenizer.tokenize(sent)]


def bigrams(tokens):
    lowered = [token.lower() for token in tokens]
    return set(zip(lowered, lowered[1:]))


def example_stats(pair):
    document, summary = pair
    doc_sents, sum_sents = split_sentences(document), split_sentences(summary)
    doc_tokens, sum_tokens = tokenize(doc_sents), tokenize(sum_sents)

    sum_bigrams = bigrams(sum_tokens)
    novel_bigram_ratio = len(sum_bigrams - bigrams(doc_tokens)) / len(sum_bigrams) if sum_bigrams else None
    return len(doc_sents), len(doc_tokens), len(sum_sents), len(sum_tokens), novel_bigram_ratio


def compute_split_totals(pool, split_dir, desc):
    totals = dict(n=0, doc_sents=0, doc_words=0, sum_sents=0, sum_words=0, novel_sum=0.0, novel_n=0)
    results = pool.imap_unordered(example_stats, iter_split(split_dir), chunksize=CHUNK_SIZE)
    for doc_sents, doc_words, sum_sents, sum_words, novel_ratio in tqdm(results, desc=desc, unit=" ex"):
        totals["n"] += 1
        totals["doc_sents"] += doc_sents
        totals["doc_words"] += doc_words
        totals["sum_sents"] += sum_sents
        totals["sum_words"] += sum_words
        if novel_ratio is not None:
            totals["novel_sum"] += novel_ratio
            totals["novel_n"] += 1
    return totals


def averages(totals):
    n = totals["n"]
    return {
        "doc_words": totals["doc_words"] / n,
        "doc_sents": totals["doc_sents"] / n,
        "sum_words": totals["sum_words"] / n,
        "sum_sents": totals["sum_sents"] / n,
        "novel_bigrams": 100 * totals["novel_sum"] / totals["novel_n"],
    }


def to_latex(results):
    def row(label, values):
        return f"{label:<30}& " + " & ".join(f"{value:<13}" for value in values) + r" \\"

    names = list(results)
    stats = [results[name]["all"] for name in names]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(names) + "}",
        r"\toprule",
        " & " + " & ".join(rf"\textbf{{{name}}}" for name in names) + r" \\",
        r"\midrule",
        row("Domain", [DATASETS[name]["domain"] for name in names]),
        row("Train", [f"{results[name]['sizes']['train']:,}" for name in names]),
        row("Validation", [f"{results[name]['sizes']['validation']:,}" for name in names]),
        row("Test", [f"{results[name]['sizes']['test']:,}" for name in names]),
        row(r"Avg.\ doc words", [f"{s['doc_words']:.2f}" for s in stats]),
        row(r"Avg.\ doc sentences", [f"{s['doc_sents']:.2f}" for s in stats]),
        row(r"Avg.\ summary words", [f"{s['sum_words']:.2f}" for s in stats]),
        row(r"Avg.\ summary sentences", [f"{s['sum_sents']:.2f}" for s in stats]),
        row(r"Novel bigrams in summary (\%)", [f"{s['novel_bigrams']:.2f}" for s in stats]),
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Statistics of the summarization datasets used for fine-tuning. Averages are computed over all "
        r"splits with NLTK tokenization (punctuation tokens included); training and validation summaries are the "
        r"entity-corrected versions produced by the corrector.}",
        r"\label{tab:datasets}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    results = {}
    with Pool(NUM_WORKERS) as pool:
        for name, info in DATASETS.items():
            split_totals = {
                split: compute_split_totals(pool, Path(info["path"]) / split, desc=f"{name}/{split}")
                for split in SPLITS
            }
            all_totals = {key: sum(totals[key] for totals in split_totals.values()) for key in split_totals["train"]}
            results[name] = {
                "sizes": {split: totals["n"] for split, totals in split_totals.items()},
                **{split: averages(totals) for split, totals in split_totals.items()},
                "all": averages(all_totals),
            }

    header = f"{'':<14}{'n':>9}{'doc words':>11}{'doc sents':>11}{'sum words':>11}{'sum sents':>11}{'novel bi %':>12}"
    for name, result in results.items():
        print(f"\n===== {name} =====")
        print(header)
        for split in SPLITS + ["all"]:
            n = sum(result["sizes"].values()) if split == "all" else result["sizes"][split]
            s = result[split]
            print(f"{split:<14}{n:>9,}{s['doc_words']:>11.2f}{s['doc_sents']:>11.2f}"
                  f"{s['sum_words']:>11.2f}{s['sum_sents']:>11.2f}{s['novel_bigrams']:>12.2f}")

    print("\n" + to_latex(results))


if __name__ == "__main__":
    main()

import argparse
from rouge_score import rouge_scorer
import nltk
from datasets import load_dataset
import numpy as np


USE_SMALLER_SUBSET = True    # MODIFY HERE TO USE A SMALLER SUBSET OF THE DATASET
SUBSET_LOWER_LIMIT = 1000000
SUBSET_UPPER_LIMIT = 4000000


parser = argparse.ArgumentParser()

parser.add_argument("--c4_split", type=str, default="realnewslike", choices=["en","realnewslike"])
parser.add_argument("--rouge_type", type=str, default="rouge1", choices=["rouge1","rouge2","rougeL"])
parser.add_argument("--topk", type=int, default=5)

args = parser.parse_args()

OUTPUT_PATH = "c4_{}_processed_ROUGE_1_to_4_mil".format(args.c4_split)


mask_token = "<mask>"

scorer = rouge_scorer.RougeScorer([args.rouge_type])


def calc_rouge_score_and_select_top_k(example):
    sentences = nltk.sent_tokenize(example["text"])
    
    scores = []
    for i, sent in enumerate(sentences):
        summ = sent
        doc = " ".join([s for j,s in enumerate(sentences) if i !=j])
        score = scorer.score(doc, summ)
        scores.append(score[args.rouge_type].fmeasure)

    # top k
    if len(scores) <= args.topk:
        ind = np.arange(len(scores))
    else:
        ind = np.argpartition(scores, -args.topk)[-args.topk:]
    
    example["documents"] = [" ".join([s if j!=i else mask_token for j,s in enumerate(sentences)]) for i in ind ]
    example["summaries"] = [ sentences[i] for i in ind ]
    example["rouge"] = [ scores[i] for i in ind ]

    return example


dataset = load_dataset("c4",args.c4_split, cache_dir="./cache")

dataset.pop("validation")

if USE_SMALLER_SUBSET:
    dataset["train"] = dataset["train"].select(list(range(SUBSET_LOWER_LIMIT, SUBSET_UPPER_LIMIT)))

dataset["train"] = dataset["train"].map(
    calc_rouge_score_and_select_top_k,
    remove_columns=["url","text","timestamp"],
    batched=False,
    num_proc=16,
    keep_in_memory=True
)

dataset.save_to_disk(OUTPUT_PATH)

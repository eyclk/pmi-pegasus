import argparse
#  from rouge_score import rouge_scorer
import nltk
from datasets import load_dataset
import numpy as np
from calc_pmi_for_a_document import calculate_pmi

parser = argparse.ArgumentParser()

parser.add_argument("--c4_split", type=str, default="realnewslike", choices=["en","realnewslike"])
#  parser.add_argument("--rouge_type", type=str, default="rouge1", choices=["rouge1","rouge2","rougeL"])
parser.add_argument("--topk", type=int, default=5)

args = parser.parse_args()

mask_token = "<mask>"

#  scorer = rouge_scorer.RougeScorer([args.rouge_type])
    
def calc_pmi_score_and_select_top_k(example):
    sentences = nltk.sent_tokenize(example["text"])
    
    scores = []
    for i, sent in enumerate(sentences):
        summ = sent
        doc = " ".join([s for j,s in enumerate(sentences) if i !=j])
        pmi_score = calculate_pmi(summ, doc)

        #  scores.append(pmi_score[args.rouge_type].fmeasure)
        scores.append(pmi_score)

    # top k
    if len(scores) <= args.topk:
        ind = np.arange(len(scores))
    else:
        ind = np.argpartition(scores, -args.topk)[-args.topk:]
    
    example["documents"] = [" ".join([s if j!=i else mask_token for j,s in enumerate(sentences)]) for i in ind ]
    example["summaries"] = [ sentences[i] for i in ind ]
    example["pmi"] = [ scores[i] for i in ind ]  #  ["rouge"]

    return example

dataset = load_dataset("c4",args.c4_split, cache_dir="./cache")

dataset.pop("validation")
dataset["train"] = dataset["train"].select(list(range(1000)))
dataset["train"] = dataset["train"].map(
    calc_pmi_score_and_select_top_k,
    remove_columns=["url","text","timestamp"],
    batched=False,
    num_proc=16,
    keep_in_memory=True
)

dataset.save_to_disk("c4_{}_processed_with_PMI".format(args.c4_split))

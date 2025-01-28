import numpy as np
from datasets import load_from_disk
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("--processed_data", type=str, default="c4_realnewslike_processed")
parser.add_argument("--topk", type=int, default=5)
#  parser.add_argument("--factcc_pred", type=str, default="scripts/factcc_dummy.json")

args = parser.parse_args()

# flip score generated by factcc
#  scores = [[1 if x== 0 else 0  for x in pred] for pred in json.load(open(args.factcc_pred))]

dataset = load_from_disk(args.processed_data)

#  dataset["train"] = dataset["train"].add_column(name="factcc", column=scores)

def combine_scores(example):
    #  scores = [r+f for r,f in zip(example["rouge"],example["factcc"])]
    scores = [r for r in example["rouge"]]

    am = np.argmax(scores)
    example["document"] = example["documents"][am]
    example["summary"] = example["summaries"][am]
    return example


dataset["train"] = dataset["train"].map(
    combine_scores,
    remove_columns=["rouge","documents","summaries"],  # remove_columns=["rouge","factcc","documents","summaries"],
    keep_in_memory=True,
    num_proc=16,
)

dataset.save_to_disk("{}_combined_No_factcc_only_rouge".format(args.processed_data))

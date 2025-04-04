import argparse
#  from rouge_score import rouge_scorer
import nltk
from datasets import load_dataset
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import math
import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--c4_split", type=str, default="realnewslike", choices=["en", "realnewslike"])
#  parser.add_argument("--rouge_type", type=str, default="rouge1", choices=["rouge1","rouge2","rougeL"])
parser.add_argument("--topk", type=int, default=5)

args = parser.parse_args()

mask_token = "<mask>"

#  scorer = rouge_scorer.RougeScorer([args.rouge_type])

# Load pre-trained model tokenizer (vocabulary)
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Load pre-trained model (weights)
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

all_PMI_scores_dict = {}


def conditional_probability(target_sentence, context_sentence):
    # Tokenize the context sentence
    context_inputs = tokenizer(context_sentence, return_tensors='pt', truncation=True, max_length=512).to(device)

    # Tokenize the target sentence
    target_inputs = tokenizer(target_sentence, return_tensors='pt', truncation=True, max_length=512).to(device)

    losses = model(context_inputs['input_ids'], labels=target_inputs['input_ids'])
    # print("Probability of target sentence given context sentence:", losses.loss.item())
    return math.exp(-losses.loss.item())


def marginal_probability(context_sentence):
    # Tokenize the context sentence
    context_inputs = tokenizer(context_sentence, return_tensors='pt', truncation=True, max_length=512).to(device)

    # Generate output
    output = model.generate(context_inputs['input_ids'], max_new_tokens=40, num_beams=5, no_repeat_ngram_size=2,
                            early_stopping=True)

    losses = model(context_inputs['input_ids'], labels=output)

    # Print probability
    # print("Probability of target sentence given context sentence:", p_x_given_y)
    return math.exp(-losses.loss.item())


def calculate_pmi(target_sentence, doc_without_target_sentence):
    p_x_given_y = conditional_probability(target_sentence, doc_without_target_sentence)
    p_x = marginal_probability(doc_without_target_sentence)

    pmi = math.log2(p_x_given_y / p_x)
    return pmi


def calc_pmi_for_all(training_dataset):
    # global model
    for t in tqdm.tqdm(training_dataset):
        temp_text = t["text"]
        sentences = nltk.sent_tokenize(temp_text)

        temp_scores = []
        for i, sent in enumerate(sentences):
            summ = sent
            doc = " ".join([s for j, s in enumerate(sentences) if i != j])
            pmi_score = calculate_pmi(summ, doc)
            temp_scores.append(pmi_score)
        all_PMI_scores_dict[temp_text] = temp_scores


def calc_pmi_score_and_select_top_k(example):
    temp_text = example["text"]
    sentences = nltk.sent_tokenize(temp_text)

    """scores = []
    for i, sent in enumerate(sentences):
        summ = sent
        doc = " ".join([s for j,s in enumerate(sentences) if i !=j])
        pmi_score = calculate_pmi(summ, doc)
        #  scores.append(pmi_score[args.rouge_type].fmeasure)
        scores.append(pmi_score)"""
    scores = all_PMI_scores_dict[temp_text]

    # top k
    if len(scores) <= args.topk:
        ind = np.arange(len(scores))
    else:
        ind = np.argpartition(scores, -args.topk)[-args.topk:]

    example["documents"] = [" ".join([s if j != i else mask_token for j, s in enumerate(sentences)]) for i in ind]
    example["summaries"] = [sentences[i] for i in ind]
    example["pmi"] = [scores[i] for i in ind]  # ["rouge"]

    return example


dataset = load_dataset("c4", args.c4_split, cache_dir="./cache")

dataset.pop("validation")
dataset["train"] = dataset["train"].select(list(range(1000)))

calc_pmi_for_all(dataset["train"])

dataset["train"] = dataset["train"].map(
    calc_pmi_score_and_select_top_k,
    remove_columns=["url", "text", "timestamp"],
    batched=False,
    num_proc=16,
    keep_in_memory=True
)

dataset.save_to_disk("c4_{}_processed_PMI".format(args.c4_split))

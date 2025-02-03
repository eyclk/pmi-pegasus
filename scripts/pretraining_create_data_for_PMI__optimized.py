import argparse
#  from rouge_score import rouge_scorer
import nltk
from datasets import load_dataset
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import math
import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


MAX_WORKERS = 11

USE_SMALLER_SUBSET = True
SUBSET_LIMIT = 1000


multiprocessing.set_start_method('spawn', force=True)

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

model.eval()

all_PMI_scores_dict = {}



def conditional_probability(target_sentence, context_sentence):
    # Tokenize the context sentence
    context_inputs = tokenizer(context_sentence, return_tensors='pt', truncation=True, max_length=512).to(device)

    # Tokenize the target sentence
    target_inputs = tokenizer(target_sentence, return_tensors='pt', truncation=True, max_length=512).to(device)

    with torch.no_grad():
        losses = model(context_inputs['input_ids'], labels=target_inputs['input_ids'])
    # print("Probability of target sentence given context sentence:", losses.loss.item())
    return math.exp(-losses.loss.item())


def marginal_probability(context_sentence):
    # Tokenize the context sentence
    context_inputs = tokenizer(context_sentence, return_tensors='pt', truncation=True, max_length=512).to(device)

    # Generate output  -------------------> I SHOULD CHECK WHETHER THE GENERATION PARAMETERS CAN BE IMPROVED
    output = model.generate(context_inputs['input_ids'], max_new_tokens=40, num_beams=5, no_repeat_ngram_size=2,
                            early_stopping=True)

    with torch.no_grad():
        losses = model(context_inputs['input_ids'], labels=output)

    # Print probability
    # print("Probability of target sentence given context sentence:", p_x_given_y)
    return math.exp(-losses.loss.item())


def calculate_pmi(target_sentence, doc_without_target_sentence):
    p_x_given_y = conditional_probability(target_sentence, doc_without_target_sentence)
    p_x = marginal_probability(doc_without_target_sentence)

    pmi = math.log2(p_x_given_y / p_x)
    return pmi


"""def calc_pmi_for_all(training_dataset):
    # global model
    def process_text(t):
        temp_text = t["text"]
        sentences = nltk.sent_tokenize(temp_text)

        temp_scores = []
        for i, sent in enumerate(sentences):
            summ = sent
            doc = temp_text.replace(sent, "", 1)
            # If replace function fails, use the other approach with join function. This should not happen, but this check is placed here just in case.
            if doc == temp_text:
                doc = " ".join([s for j, s in enumerate(sentences) if i != j])

            pmi_score = calculate_pmi(summ, doc)
            temp_scores.append(pmi_score)
        return temp_text, temp_scores

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm.tqdm(executor.map(process_text, training_dataset), total=len(training_dataset)))

    for text, scores in results:
        all_PMI_scores_dict[text] = scores"""


def process_text(t):
    temp_text = t["text"]
    sentences = nltk.sent_tokenize(temp_text)

    temp_scores = []
    for i, sent in enumerate(sentences):
        summ = sent
        doc = temp_text.replace(sent, "", 1)
        # If replace function fails, use the other approach with join function. This should not happen, but this check is placed here just in case.
        if doc == temp_text:
            doc = " ".join([s for j, s in enumerate(sentences) if i != j])

        pmi_score = calculate_pmi(summ, doc)
        temp_scores.append(pmi_score)
    return temp_text, temp_scores

def calc_pmi_for_all(training_dataset):
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm.tqdm(executor.map(process_text, training_dataset), total=len(training_dataset)))

    for text, scores in results:
        all_PMI_scores_dict[text] = scores


def calc_pmi_score_and_select_top_k(example):
    temp_text = example["text"]
    sentences = nltk.sent_tokenize(temp_text)

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


if __name__ == "__main__":

    dataset = load_dataset("c4", args.c4_split, cache_dir="./cache")

    dataset.pop("validation")

    if USE_SMALLER_SUBSET:
        dataset["train"] = dataset["train"].select(list(range(SUBSET_LIMIT)))

    calc_pmi_for_all(dataset["train"])

    dataset["train"] = dataset["train"].map(
        calc_pmi_score_and_select_top_k,
        remove_columns=["url", "text", "timestamp"],
        batched=False,
        num_proc=16,
        keep_in_memory=True
    )

    dataset.save_to_disk("c4_{}_processed_with_PMI".format(args.c4_split))

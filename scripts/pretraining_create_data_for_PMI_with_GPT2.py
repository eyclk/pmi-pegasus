import argparse
import nltk
from datasets import load_dataset
import numpy as np
import torch
import math
import tqdm
import torch.nn.functional as F
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import BartTokenizer, BartForConditionalGeneration
# from fastT5 import export_and_get_onnx_model
# from transformers import T5Tokenizer

from transformers import AutoModelForCausalLM, GPT2Tokenizer


SENTENCE_BATCH_SIZE = 32

USE_SMALLER_SUBSET = True    # MODIFY HERE TO USE A SMALLER SUBSET OF THE DATASET
SUBSET_LIMIT = 1000


parser = argparse.ArgumentParser()

parser.add_argument("--c4_split", type=str, default="realnewslike", choices=["en", "realnewslike"])
#  parser.add_argument("--rouge_type", type=str, default="rouge1", choices=["rouge1","rouge2","rougeL"])
parser.add_argument("--topk", type=int, default=5)

args = parser.parse_args()

mask_token = "<mask>"

# Load pre-trained model tokenizer (vocabulary)
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# Load pre-trained model (weights)
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-base', forced_bos_token_id=0)


model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Ensure left-padding for GPT-2

"""model_name = 't5-base'
model = export_and_get_onnx_model(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)"""


# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.eval()

all_PMI_scores_dict = {}


def conditional_probability(context_inputs, target_inputs):
    # batch_size = context_inputs['input_ids'].shape[0]

    combined_inputs = torch.cat([context_inputs['input_ids'], target_inputs['input_ids']], dim=1).to(device)

    # Create attention mask
    combined_attention_mask = torch.cat([context_inputs['attention_mask'], target_inputs['attention_mask']], dim=1).to(
        device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=combined_inputs, attention_mask=combined_attention_mask)

    logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

    # Shift the target labels so that loss is calculated only on target tokens
    labels = target_inputs["input_ids"].to(device)

    # Ensure padding tokens (-100) are ignored
    labels[labels == tokenizer.pad_token_id] = -100

    loss_per_token = F.cross_entropy(
        logits[:, -labels.shape[1]:, :].contiguous().view(-1, model.config.vocab_size),
        labels.view(-1),
        ignore_index=-100,
        reduction="none"
    )

    loss_per_token = loss_per_token.view(labels.shape)

    valid_token_mask = (labels != -100).float()
    loss_per_example = loss_per_token.sum(dim=1) / valid_token_mask.sum(dim=1)

    conditional_probabilities_per_example = [math.exp(-l.item()) for l in loss_per_example]

    return conditional_probabilities_per_example


def marginal_probability(context_inputs):
    generated_output = model.generate(
        input_ids=context_inputs['input_ids'],
        attention_mask=context_inputs['attention_mask'],
        max_new_tokens=40,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    # Ensure generated_output has the correct shape
    generated_output = generated_output[:, context_inputs['input_ids'].shape[1]:]

    # Concatenate context and generated text
    combined_inputs = torch.cat([context_inputs['input_ids'], generated_output], dim=1)

    # Fix attention mask to match combined input
    combined_attention_mask = torch.cat(
        [context_inputs['attention_mask'], torch.ones_like(generated_output, device=device)], dim=1
    )

    """max_length = min(1024, combined_inputs.shape[1])  # Ensure max length does not exceed 1024
    combined_inputs = combined_inputs[:, :max_length]
    combined_attention_mask = combined_attention_mask[:, :max_length]"""

    # Ensure shape consistency
    assert combined_inputs.shape == combined_attention_mask.shape, "Mismatch in input and attention mask size!"

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=combined_inputs, attention_mask=combined_attention_mask)

    logits = outputs.logits

    # Extract generated portion
    labels = generated_output.to(device)
    labels[labels == tokenizer.pad_token_id] = -100

    loss_per_token = F.cross_entropy(
        logits[:, -labels.shape[1]:, :].contiguous().view(-1, model.config.vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none"
    )

    loss_per_token = loss_per_token.view(labels.shape)
    valid_token_mask = (labels != -100).float()
    loss_per_example = loss_per_token.sum(dim=1) / valid_token_mask.sum(dim=1)

    return [math.exp(-l.item()) for l in loss_per_example]


def calculate_pmi(target_sentences_per_text, docs_without_target_sentences_per_text):
    context_inputs = tokenizer(docs_without_target_sentences_per_text, return_tensors='pt', padding=True, add_special_tokens=False,
                               truncation=True, max_length=512).to(device)
    target_inputs = tokenizer(target_sentences_per_text, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False,
                              max_length=512).to(device)

    p_x_given_y_list = conditional_probability(context_inputs, target_inputs)
    p_x_list = marginal_probability(context_inputs)

    """p_x_given_y_list = conditional_probability(target_sentences_per_text, docs_without_target_sentences_per_text)
    p_x_list = marginal_probability(docs_without_target_sentences_per_text)"""

    pmi_list = [math.log2(a / b) for a, b in zip(p_x_given_y_list, p_x_list)]

    return pmi_list


def single_process_calc_pmi_for_all(training_dataset):

    all_sentences = []
    texts_corresponding_to_sentences = []
    for s in tqdm.tqdm(training_dataset, desc="Splitting all texts into separate sentences"):
        temp_text = s["text"]
        sentences = nltk.sent_tokenize(temp_text)
        all_sentences.extend(sentences)
        texts_corresponding_to_sentences.extend([temp_text] * len(sentences))

    # Batch all sentences according to SENTENCE_BATCH_SIZE and send them to calculate_pmi function
    for i in tqdm.tqdm(range(0, len(all_sentences), SENTENCE_BATCH_SIZE), desc="Calculating PMI"):
        sentences_per_batch = all_sentences[i : i+SENTENCE_BATCH_SIZE]
        docs_per_batch = [t.replace(s, "", 1) for t, s in zip(texts_corresponding_to_sentences[i : i + SENTENCE_BATCH_SIZE], sentences_per_batch)]
        pmi_scores_per_batch = calculate_pmi(sentences_per_batch, docs_per_batch)

        for score, text in zip(pmi_scores_per_batch, texts_corresponding_to_sentences[i : i + SENTENCE_BATCH_SIZE]):
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

    example["documents"] = [" ".join([s if j != i else mask_token for j, s in enumerate(sentences)]) for i in ind]
    example["summaries"] = [sentences[i] for i in ind]
    example["pmi"] = [scores[i] for i in ind]

    return example


if __name__ == "__main__":

    dataset = load_dataset("c4", args.c4_split, cache_dir="./cache")

    dataset.pop("validation")

    if USE_SMALLER_SUBSET:
        dataset["train"] = dataset["train"].select(list(range(SUBSET_LIMIT)))

    single_process_calc_pmi_for_all(dataset["train"])

    dataset["train"] = dataset["train"].map(
        calc_pmi_score_and_select_top_k,
        remove_columns=["url", "text", "timestamp"],
        batched=False,
        num_proc=16,
        keep_in_memory=True
    )

    dataset.save_to_disk("c4_{}_processed_PMI".format(args.c4_split))

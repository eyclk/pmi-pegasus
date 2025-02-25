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

# from transformers import AutoModelForCausalLM, AutoTokenizer

# from fastT5 import export_and_get_onnx_model
# from transformers import T5Tokenizer

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer


SENTENCE_BATCH_SIZE = 64

USE_SMALLER_SUBSET = True    # MODIFY HERE TO USE A SMALLER SUBSET OF THE DATASET
SUBSET_LIMIT = 1000


parser = argparse.ArgumentParser()

parser.add_argument("--c4_split", type=str, default="realnewslike", choices=["en", "realnewslike"])
#  parser.add_argument("--rouge_type", type=str, default="rouge1", choices=["rouge1","rouge2","rougeL"])
parser.add_argument("--topk", type=int, default=5)

args = parser.parse_args()

mask_token = "<mask>"

# Load pre-trained model tokenizer (vocabulary)
# tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load pre-trained model (weights)
# model = T5ForConditionalGeneration.from_pretrained('t5-small')


model_name = "deepseek-ai/DeepSeek-V2-Lite"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# model = AutoModelForCausalLM.from_pretrained("gpt2")
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token

"""model_name = 't5-base'
model = export_and_get_onnx_model(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)"""

# tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')   # google/pegasus-xsum
# model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large')   # google/pegasus-xsum


# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.eval()

all_PMI_scores_dict = {}


def conditional_probability(context_inputs, target_inputs):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(context_inputs['input_ids'], labels=target_inputs['input_ids'], return_dict=True)

        logits = outputs.logits

        labels_shifted = target_inputs["input_ids"][:, 1:].contiguous()
        logits_shifted = logits[:, :-1, :].contiguous()

        loss_per_token = F.cross_entropy(
            logits_shifted.view(-1, model.config.vocab_size),
            labels_shifted.view(-1),
            ignore_index=-100,
            reduction="none"
        )

        loss_per_token = loss_per_token.view(labels_shifted.shape)

        valid_token_mask = (labels_shifted != -100).float()
        loss_per_example = loss_per_token.sum(dim=1) / valid_token_mask.sum(dim=1)

        conditional_probabilities_per_example = [math.exp(-l.item()) for l in loss_per_example]

    return conditional_probabilities_per_example


def marginal_probability(context_inputs):
    generated_output = model.generate(
        context_inputs['input_ids'],
        max_new_tokens=40,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    """generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in generated_output]

        generated_labels = tokenizer(
            generated_texts, return_tensors='pt', padding=True, truncation=True, max_length=512
        ).to(device)

        # Ensure padding tokens (-100) are ignored in loss computation
        generated_labels["input_ids"][generated_labels["input_ids"] == tokenizer.pad_token_id] = -100"""

    generated_output[generated_output == tokenizer.pad_token_id] = -100

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=context_inputs['input_ids'], labels=generated_output)

        logits = outputs.logits

        labels_shifted = generated_output[:, 1:].contiguous()
        logits_shifted = logits[:, :-1, :].contiguous()

        loss_per_token = F.cross_entropy(
            logits_shifted.view(-1, model.config.vocab_size),
            labels_shifted.view(-1),
            ignore_index=-100,
            reduction="none"
        )

        loss_per_token = loss_per_token.view(labels_shifted.shape)

        valid_token_mask = (labels_shifted != -100).float()
        loss_per_example = loss_per_token.sum(dim=1) / valid_token_mask.sum(dim=1)

        marginal_probabilities_per_example = [math.exp(-l.item()) for l in loss_per_example]

    return marginal_probabilities_per_example


def calculate_pmi(target_sentences_per_text, docs_without_target_sentences_per_text):
    context_inputs = tokenizer(docs_without_target_sentences_per_text, return_tensors='pt', padding=True,
                               truncation=True, max_length=512).to(device)
    target_inputs = tokenizer(target_sentences_per_text, return_tensors='pt', padding=True, truncation=True,
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

import argparse
import nltk
from datasets import load_dataset
# import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
# import math
import tqdm
# import torch.nn.functional as F


SENTENCE_BATCH_SIZE = 48

# Transfer to CPU every N batches
N = 55

logits_and_labels_save_location = "./scripts/logits_and_labels/limited_test_1000_rows/"

USE_SMALLER_SUBSET = True    # MODIFY HERE TO USE A SMALLER SUBSET OF THE DATASET
SUBSET_LIMIT = 1000


parser = argparse.ArgumentParser()

parser.add_argument("--c4_split", type=str, default="realnewslike", choices=["en", "realnewslike"])
#  parser.add_argument("--rouge_type", type=str, default="rouge1", choices=["rouge1","rouge2","rougeL"])
parser.add_argument("--topk", type=int, default=5)

args = parser.parse_args()

mask_token = "<mask>"

# Load pre-trained model tokenizer (vocabulary)
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Load pre-trained model (weights)
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.eval()

all_PMI_scores_dict = {}


def logits_and_labels_for_conditional_probability(target_sentences, context_sentences):
    # Tokenize the context sentence
    context_inputs = tokenizer(context_sentences, return_tensors='pt', padding=True, truncation=True,
                               max_length=512).to(device)

    # Tokenize the target sentence
    target_inputs = tokenizer(target_sentences, return_tensors='pt', padding=True, truncation=True, max_length=512).to(
        device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(context_inputs['input_ids'], labels=target_inputs['input_ids'],
                        return_dict=True)

    #   print("\n\n", outputs.loss, "\n\n")

    # Extract logits
    """logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

    # Shift labels left to align with logits
    labels_shifted = target_inputs["input_ids"][:, 1:].contiguous()
    logits_shifted = logits[:, :-1, :].contiguous()

    # Step 5: Compute Per-Token Loss Correctly
    loss_per_token = F.cross_entropy(
        logits_shifted.view(-1, model.config.vocab_size),
        labels_shifted.view(-1),
        ignore_index=-100,
        reduction="none"
    )

    # Reshape back to (batch_size, seq_length - 1)
    loss_per_token = loss_per_token.view(labels_shifted.shape)

    # Mask out padding (-100) and count valid tokens per example
    valid_token_mask = (labels_shifted != -100).float()
    valid_token_count_per_example = valid_token_mask.sum(dim=1)  # Count valid tokens per example

    # Step 6: Compute Corrected Per-Example Loss
    loss_per_example = loss_per_token.sum(dim=1) / valid_token_count_per_example"""

    # Print per-example loss
    """for i in loss_per_example:
        print(f"Loss: {i.item()}\n")

    # Optional: Compute batch average loss (matches model.loss)
    # batch_avg_loss = loss_per_example.mean().item()
    print(f"\nTotal batch loss (averaged): {batch_avg_loss}")"""
    # return [math.exp(-l.item()) for l in loss_per_example]

    return outputs.logits, target_inputs["input_ids"]


def logits_and_labels_for_marginal_probability(context_sentences):
    # Step 1: Tokenize the context sentence (input)
    context_inputs = tokenizer(
        context_sentences, return_tensors='pt', padding=True, truncation=True, max_length=512
    ).to(device)

    # Step 2: Generate output sentences
    generated_output = model.generate(
        context_inputs['input_ids'],
        max_new_tokens=40,
        num_beams=4,   #  5 --> 4
        no_repeat_ngram_size=3,  # 2 --> 3
        early_stopping=True
    )

    # Step 3: Decode & Tokenize the Generated Sentences Properly
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in generated_output]

    generated_labels = tokenizer(
        generated_texts, return_tensors='pt', padding=True, truncation=True, max_length=512
    ).to(device)

    # Ensure padding tokens (-100) are ignored in loss computation
    generated_labels["input_ids"][generated_labels["input_ids"] == tokenizer.pad_token_id] = -100

    # Step 4: Forward pass with context_inputs and generated_labels
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(input_ids=context_inputs['input_ids'], labels=generated_labels['input_ids'])

    # print("\n\n", outputs.loss, "\n\n")

    # Extract logits
    """logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

    # Shift labels left to align with logits
    labels_shifted = generated_labels["input_ids"][:, 1:].contiguous()
    logits_shifted = logits[:, :-1, :].contiguous()

    # Step 5: Compute Per-Token Loss Correctly
    loss_per_token = F.cross_entropy(
        logits_shifted.view(-1, model.config.vocab_size),
        labels_shifted.view(-1),
        ignore_index=-100,
        reduction="none"
    )

    # Reshape back to (batch_size, seq_length - 1)
    loss_per_token = loss_per_token.view(labels_shifted.shape)

    # Mask out padding (-100) and count valid tokens per example
    valid_token_mask = (labels_shifted != -100).float()
    valid_token_count_per_example = valid_token_mask.sum(dim=1)  # Count valid tokens per example

    # Step 6: Compute Corrected Per-Example Loss
    loss_per_example = loss_per_token.sum(dim=1) / valid_token_count_per_example"""


    # Print per-example loss
    """for i in loss_per_example:
        print(f"Loss: {i.item()}\n")

    # Optional: Compute batch average loss (matches model.loss)
    batch_avg_loss = loss_per_example.mean().item()
    print(f"\nTotal batch loss (averaged): {batch_avg_loss}")"""
    # return [math.exp(-l.item()) for l in loss_per_example]

    return outputs.logits, generated_labels["input_ids"]


def save_logits_and_labels(target_sentences_per_text, docs_without_target_sentences_per_text):
    cond_logits, cond_labels = logits_and_labels_for_conditional_probability(target_sentences_per_text, docs_without_target_sentences_per_text)
    marg_logits, marg_labels = logits_and_labels_for_marginal_probability(docs_without_target_sentences_per_text)

    # pmi_list = [math.log2(a / b) for a, b in zip(p_x_given_y_list, p_x_list)]
    # return pmi_list

    # Save the logits and labels
    """torch.save(cond_logits, logits_and_labels_save_location + "cond_logits.pt")
    torch.save(cond_labels, logits_and_labels_save_location + "cond_labels.pt")

    torch.save(marg_logits, logits_and_labels_save_location + "marg_logits.pt")
    torch.save(marg_labels, logits_and_labels_save_location + "marg_labels.pt")"""

    return cond_logits, cond_labels, marg_logits, marg_labels


def acquire_logits_from_forward_passes(training_dataset):

    all_sentences = []
    texts_corresponding_to_sentences = []
    for s in tqdm.tqdm(training_dataset, desc="Splitting all texts into separate sentences"):
        temp_text = s["text"]
        sentences = nltk.sent_tokenize(temp_text)
        all_sentences.extend(sentences)
        texts_corresponding_to_sentences.extend([temp_text] * len(sentences))

    # Create lists to store all logits and labels
    all_cond_logits = []
    all_cond_labels = []
    all_marg_logits = []
    all_marg_labels = []

    # Batch all sentences according to SENTENCE_BATCH_SIZE and send them to calculate_pmi function
    for i in tqdm.tqdm(range(0, len(all_sentences), SENTENCE_BATCH_SIZE), desc="Calculating PMI"):
        sentences_per_batch = all_sentences[i : i+SENTENCE_BATCH_SIZE]
        docs_per_batch = [t.replace(s, "", 1) for t, s in zip(texts_corresponding_to_sentences[i : i + SENTENCE_BATCH_SIZE], sentences_per_batch)]

        cond_logits, cond_labels, marg_logits, marg_labels = save_logits_and_labels(sentences_per_batch, docs_per_batch)

        """all_cond_logits.append(cond_logits.to("cpu"))   #  AN EXTREMELY COSTLY OPERATION !!!!!!!!!!!!!!!!!!
        all_cond_labels.append(cond_labels.to("cpu"))   #  AN EXTREMELY COSTLY OPERATION !!!!!!!!!!!!!!!!!!
        all_marg_logits.append(marg_logits.to("cpu"))   #  AN EXTREMELY COSTLY OPERATION !!!!!!!!!!!!!!!!!!
        all_marg_labels.append(marg_labels.to("cpu"))   #  AN EXTREMELY COSTLY OPERATION !!!!!!!!!!!!!!!!!!"""

        all_cond_logits.append(cond_logits)  # Keep on GPU for a while
        all_cond_labels.append(cond_labels)  # Keep on GPU for a while
        all_marg_logits.append(marg_logits)  # Keep on GPU for a while
        all_marg_labels.append(marg_labels)  # Keep on GPU for a while

        if (i // SENTENCE_BATCH_SIZE) % N == 0:  # Transfer to CPU every N batches
            all_cond_logits = [logits.to("cpu") for logits in all_cond_logits]
            all_cond_labels = [labels.to("cpu") for labels in all_cond_labels]
            all_marg_logits = [logits.to("cpu") for logits in all_marg_logits]
            all_marg_labels = [labels.to("cpu") for labels in all_marg_labels]

        """for score, text in zip(pmi_scores_per_batch, texts_corresponding_to_sentences[i : i + SENTENCE_BATCH_SIZE]):
            if all_PMI_scores_dict.get(text) is None:
                all_PMI_scores_dict[text] = [score]
            else:
                all_PMI_scores_dict[text].append(score)"""

    torch.save(all_cond_logits, logits_and_labels_save_location + "cond_logits.pt")
    torch.save(all_cond_labels, logits_and_labels_save_location + "cond_labels.pt")

    torch.save(all_marg_logits, logits_and_labels_save_location + "marg_logits.pt")
    torch.save(all_marg_labels, logits_and_labels_save_location + "marg_labels.pt")


"""def calc_pmi_score_and_select_top_k(example):
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

    return example"""


if __name__ == "__main__":

    dataset = load_dataset("c4", args.c4_split, cache_dir="./cache")

    dataset.pop("validation")

    if USE_SMALLER_SUBSET:
        dataset["train"] = dataset["train"].select(list(range(SUBSET_LIMIT)))

    acquire_logits_from_forward_passes(dataset["train"])

    """dataset["train"] = dataset["train"].map(
        calc_pmi_score_and_select_top_k,
        remove_columns=["url", "text", "timestamp"],
        batched=False,
        num_proc=16,
        keep_in_memory=True
    )

    dataset.save_to_disk("c4_{}_processed_PMI".format(args.c4_split))"""

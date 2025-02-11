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
# from transformers import pipeline
import torch.nn.functional as F


MAX_WORKERS = 2

SENTENCE_BATCH_SIZE = 32

USE_MULTIPROCESSING = False

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

#  pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, return_loss=True)


"""def conditional_probability(target_sentence, context_sentence):
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
    return math.exp(-losses.loss.item())"""


def conditional_probability(target_sentences, context_sentences):
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
    # print("Probability of target sentence given context sentence:", losses.loss.item())

    #   print("\n\n", outputs.loss, "\n\n")

    # Extract logits
    logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

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
    loss_per_example = loss_per_token.sum(dim=1) / valid_token_count_per_example

    # Normalize Loss to Match Forward Pass Scaling
    # total_valid_tokens = valid_token_mask.sum()  # Sum of all valid tokens in batch

    # batch_avg_loss = outputs.loss * total_valid_tokens / total_valid_tokens  # This should exactly match outputs.loss

    # Print per-example loss
    """for i in loss_per_example:
        print(f"Loss: {i.item()}\n")

    # Optional: Compute batch average loss (matches model.loss)
    # batch_avg_loss = loss_per_example.mean().item()
    print(f"\nTotal batch loss (averaged): {batch_avg_loss}")"""

    # print("\n\n", context_inputs['input_ids'].shape, " ---->> ", len(losses), "\n\n")

    #   pipe_output = pipe(context_sentences, labels=target_sentences, batch_size=16)

    #  return math.exp(-losses.loss.item())
    # return [math.exp(-p['loss'].item()) for p in outputs.loss]
    return [math.exp(-l.item()) for l in loss_per_example]


def marginal_probability(context_sentences):
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
    """max_length = 100,
    min_length = 10,
    num_beams = 2,
    do_sample = True,
    top_k = 50,
    top_p = 0.9,
    no_repeat_ngram_size = 3,
    early_stopping = False,
    return_dict_in_generate = True"""

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
    logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

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

    # **Mask out padding (-100) and count valid tokens per example**
    valid_token_mask = (labels_shifted != -100).float()
    valid_token_count_per_example = valid_token_mask.sum(dim=1)  # Count valid tokens per example

    # Step 6: Compute Corrected Per-Example Loss
    loss_per_example = loss_per_token.sum(dim=1) / valid_token_count_per_example

    # Normalize Loss to Match Forward Pass Scaling
    # total_valid_tokens = valid_token_mask.sum()  # Sum of all valid tokens in batch

    # batch_avg_loss = outputs.loss * total_valid_tokens / total_valid_tokens  # This should exactly match outputs.loss

    # Print per-example loss
    """for i in loss_per_example:
        print(f"Loss: {i.item()}\n")"""

    # Optional: Compute batch average loss (matches model.loss)
    # batch_avg_loss = loss_per_example.mean().item()
    # print(f"\nTotal batch loss (averaged): {batch_avg_loss}")

    # pipe_output = pipe(context_sentences, labels=generated_output, batch_size=16)

    # Print probability
    # print("Probability of target sentence given context sentence:", p_x_given_y)

    # return math.exp(-losses.loss.item())
    # return [math.exp(-p['loss'].item()) for p in outputs.loss]

    return [math.exp(-l.item()) for l in loss_per_example]


def calculate_pmi(target_sentences_per_text, docs_without_target_sentences_per_text):
    p_x_given_y_list = conditional_probability(target_sentences_per_text, docs_without_target_sentences_per_text)
    p_x_list = marginal_probability(docs_without_target_sentences_per_text)

    pmi_list = [math.log2(a / b) for a, b in zip(p_x_given_y_list, p_x_list)]
    #  pmi = math.log2(p_x_given_y / p_x)
    return pmi_list


def process_text(t):
    temp_text = t["text"]
    sentences = nltk.sent_tokenize(temp_text)

    sentences_per_text = []
    docs_per_text = []

    # temp_scores = []
    for i, sent in enumerate(sentences):
        summ = sent
        doc = temp_text.replace(sent, "", 1)
        # If replace function fails, use the other approach with join function. This should not happen, but this check is placed here just in case.
        if doc == temp_text:
            doc = " ".join([s for j, s in enumerate(sentences) if i != j])

        # pmi_score = calculate_pmi(summ, doc)
        # temp_scores.append(pmi_score)
        sentences_per_text.append(summ)
        docs_per_text.append(doc)

    if len(sentences_per_text) > SENTENCE_BATCH_SIZE:
        # Split texts with too many paragraphs into smaller batches
        pmi_scores_per_text = []
        for i in range(0, len(sentences_per_text), SENTENCE_BATCH_SIZE):
            sentences_per_batch = sentences_per_text[i : i+SENTENCE_BATCH_SIZE]
            docs_per_batch = docs_per_text[i : i+SENTENCE_BATCH_SIZE]
            pmi_scores_per_batch = calculate_pmi(sentences_per_batch, docs_per_batch)
            pmi_scores_per_text.extend(pmi_scores_per_batch)
    else:
        pmi_scores_per_text = calculate_pmi(sentences_per_text, docs_per_text)
    return temp_text, pmi_scores_per_text  # temp_scores


def calc_pmi_for_all_multi(training_dataset):
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm.tqdm(executor.map(process_text, training_dataset), total=len(training_dataset)))

    for text, scores in results:
        all_PMI_scores_dict[text] = scores


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


    """for t in tqdm.tqdm(training_dataset):
        temp_text = t["text"]
        sentences = nltk.sent_tokenize(temp_text)

        sentences_per_text = []
        docs_per_text = []

        temp_text_list = []

        # temp_scores = []
        for i, sent in enumerate(sentences):
            summ = sent
            doc = temp_text.replace(sent, "", 1)
            # If replace function fails, use the other approach with join function. This should not happen, but this check is placed here just in case.
            if doc == temp_text:
                doc = " ".join([s for j, s in enumerate(sentences) if i != j])

            # pmi_score = calculate_pmi(summ, doc)
            # temp_scores.append(pmi_score)
            sentences_per_text.append(summ)
            docs_per_text.append(doc)

            temp_text_list.append(temp_text)

        pmi_scores_per_text = calculate_pmi(sentences_per_text, docs_per_text)
        all_PMI_scores_dict[temp_text] = pmi_scores_per_text"""


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

    if USE_MULTIPROCESSING:
        calc_pmi_for_all_multi(dataset["train"])
    else:
        single_process_calc_pmi_for_all(dataset["train"])

    dataset["train"] = dataset["train"].map(
        calc_pmi_score_and_select_top_k,
        remove_columns=["url", "text", "timestamp"],
        batched=False,
        num_proc=16,
        keep_in_memory=True
    )

    dataset.save_to_disk("c4_{}_processed_PMI".format(args.c4_split))

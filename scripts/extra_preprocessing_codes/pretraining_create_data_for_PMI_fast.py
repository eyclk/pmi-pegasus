"""Faster rewrite of scripts/pretraining_create_data_for_PMI.py that selects the same principal sentences.

Everything that decides which sentence is selected is left exactly as it is in the original script: the
same model in fp32 under autocast, the same stream of 64 sentence batches (a batch that is not full at the
end of a chunk is carried over to the next chunk rather than sent as a short batch), the same padding that
is neither masked in the attention nor excluded from the conditional probability, and the same generation
settings, including the way ties are broken. Verified against the output of the original script on two
sets: 100 short documents (100 out of 100 principal sentences and masked documents identical) and 15
documents longer than the truncation limit, where the sentence deduplication below and repeated sentences
both come into play (15 out of 15 identical).

What was made faster or leaner, none of which changes the result:

  * the documents are processed in chunks instead of splitting the whole dataset into sentences up front,
    which is what makes the original need tens of GB of RAM on the complete realnewslike split,
  * the selected sentence is kept in a numpy array indexed by example instead of a dictionary keyed by the
    document text - see the note on duplicate documents below,
  * the documents are split into sentences by a pool of worker processes instead of one,
  * the encoder runs once per batch instead of once per forward pass,
  * the beam search runs once per distinct context instead of once per sentence, which helps on documents
    that are longer than the truncation limit,
  * the principal sentence is written directly, so no separate combine script has to run afterwards: the
    combine script picks the highest scoring of the top k candidates, which is the highest scoring sentence
    of the example, and that is what is written here,
  * the mapped set is not kept in memory, so the complete split fits.

Do not expect the run to become several times shorter. Measured on a batch of 64 contexts of 512 tokens,
the beam search of the marginal probability takes 15.7 seconds while a forward pass takes 0.14 seconds, so
practically the whole run time is that one generate() call. Its throughput does not improve with a larger
batch either. Making it faster means fewer beams, fewer generated tokens or half precision, and every one
of those changes the scores - none of them is done here.

The one case where this script deliberately differs from the original: when the same document text occurs
more than once in the processed slice, the original mixes the scores of those examples together in its
dictionary and selects a sentence by an index that belongs to another example. This script scores every
example on its own, so the affected examples get their correct principal sentence instead.
"""

import argparse
import glob
import os

# The large model files are served through the Xet CDN ('us.aws.cdn.hf.co') by default, which is not
# reachable from every machine while 'huggingface.co' itself is. Falling back to the ordinary CDN keeps
# the download working there. It is set before huggingface_hub is imported, because that is when the
# setting is read, and only if it was not set explicitly in the environment already.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import math
import multiprocessing
import nltk
from datasets import load_dataset, load_from_disk
import numpy as np
import torch
import tqdm
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import BartTokenizer, BartForConditionalGeneration


# The batch size of the original script is kept, and it has to be, since the padding of a batch takes part
# in the scores it produces. Measured on a 512 token context, the throughput of the beam search that
# dominates the run is flat in the batch size anyway (4.4 sentences per
# second at a batch of 32, 64 and 128 alike), while its memory grows linearly - a batch of 128 already
# needs 6.4 GB - so there would be nothing to gain from a larger one either.
SENTENCE_BATCH_SIZE = 64
DOC_CHUNK_SIZE = 1000             # Number of documents split into sentences and scored at a time.
TOKENIZE_NUM_PROC = 16            # Worker processes that split the documents of a chunk into sentences.
MAP_NUM_PROC = 16                 # Worker processes of the final map.

MAX_SOURCE_LENGTH = 512           # Truncation limit of the original script, for the context and the target.

# Generation settings of the original script. Changing any of them changes the marginal probability and
# therefore which sentence is selected, so they are constants rather than command line arguments.
GENERATION_MAX_NEW_TOKENS = 40
GENERATION_NUM_BEAMS = 4
GENERATION_NO_REPEAT_NGRAM_SIZE = 3

USE_SMALLER_SUBSET = True    # MODIFY HERE TO USE A SMALLER SUBSET OF THE DATASET
SUBSET_LOWER_LIMIT = 0
SUBSET_UPPER_LIMIT = 1000000


parser = argparse.ArgumentParser()

parser.add_argument("--c4_split", type=str, default="realnewslike", choices=["en", "realnewslike"])
parser.add_argument("--pmi_model", type=str, default="facebook/bart-base")

# Only the highest scoring of the top k candidates ends up in the set, so k does not change the selection -
# except when two sentences of a document score exactly the same, where it decides which of them wins.
# Keep it at the value the existing dataset was built with.
parser.add_argument("--topk", type=int, default=5)

# The source documents can be taken from an already preprocessed pretraining set instead of C4 itself:
# every 'document' of such a set is the source text with the principal sentence replaced by the mask token,
# so the source text is recovered by putting the 'summary' back in place of the mask.
# CAREFUL: for PMI this does NOT give the same scores as reading C4. The context of a sentence is the raw
# document text with that sentence cut out, and the text recovered from a preprocessed set has its line
# breaks replaced by single spaces, which tokenizes differently. Use it only for a run that does not have
# to match the existing PMI dataset.
parser.add_argument("--source", type=str, default="c4", choices=["c4", "preprocessed"])
parser.add_argument("--source_dataset", type=str,
                    default="./PREPROCESSED_DATASETS/c4_realnewslike_processed_ROUGE_complete_combined")

parser.add_argument("--map_cache_dir", type=str, default="./PREPROCESSED_DATASETS/pmi_fast_map_cache",
                    help="Directory the map writes its cache files to. They are deleted once the set is saved.")

args = parser.parse_args()


# 'fast' is part of every name this script writes, so that nothing produced by the original PMI scripts
# can be overwritten. There is no separate combine step, so the set written here is already the final one
# and is named accordingly: the combine script of the other pipelines picks the highest scoring of the
# top k candidate sentences, which is simply the highest scoring sentence of the example, and that one is
# written directly here.
OUTPUT_PATH = "./PREPROCESSED_DATASETS/c4_{}_processed_PMI_fast_combined".format(args.c4_split)



mask_token = "<mask>"


def resolve_model_path(model_name_or_path):
    """Return a local directory that holds the model files.

    The old 'transformers' version of this environment cannot download from the current huggingface hub,
    so the files are fetched with 'huggingface_hub' first and are then loaded from the local snapshot.
    A path of an already downloaded model directory can be given instead, which is the way to run this
    script on a machine without access to the hub.
    """
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    try:
        return snapshot_download(model_name_or_path, allow_patterns=["*.json", "*.txt", "*.bin", "*.model"])
    except Exception as download_error:
        raise RuntimeError(
            "The model '{}' could not be downloaded from the huggingface hub: {}\n\n"
            "If this machine cannot reach the hub, download the model on a machine that can and pass the "
            "directory it was saved in with --pmi_model.".format(model_name_or_path, download_error)
        ) from download_error


model_path = resolve_model_path(args.pmi_model)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BartTokenizer.from_pretrained(model_path)

# Load pre-trained model (weights)
model = BartForConditionalGeneration.from_pretrained(model_path, forced_bos_token_id=0)

# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# The weights stay in fp32 and the forward passes run under autocast, exactly as in the original. Half
# precision weights would be faster, but they change the last digits of the scores and with them the
# occasional near-tie between two sentences.

model.eval()

# Index of the selected principal sentence of every example of the training set.
# The examples whose text contains no sentence at all keep an index of -1.
all_selected_indices = None

# Pool that splits the documents into sentences while the scores are calculated.
tokenization_pool = None


def get_text_of_example(example):
    """Return the source text of an example of the source set."""
    if args.source == "c4":
        return example["text"]

    # The mask token is put back into the document to recover the source text.
    return example["document"].replace(mask_token, example["summary"])


def get_texts_of_chunk(training_dataset, chunk_start, chunk_end):
    chunk = training_dataset[chunk_start: chunk_end]

    if args.source == "c4":
        return chunk["text"]

    return [d.replace(mask_token, s) for d, s in zip(chunk["document"], chunk["summary"])]


def encode_contexts(context_inputs):
    """Run the encoder over the contexts once, so that the two forward passes of a batch can share it.

    The original script encodes the same contexts three times per batch: once inside generate() and once
    for each of the two forward passes. Handing the encoder output to the forward passes instead is not an
    approximation, it produces bit for bit the same logits - as long as it is computed the way the forward
    passes compute it, which is without an attention mask, because the original does not pass one either.
    generate() keeps its own encoder pass, since it builds an attention mask internally and would therefore
    not accept this one without changing the generated text.
    """
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            return model.get_encoder()(input_ids=context_inputs['input_ids'], return_dict=True)


def conditional_probability(context_inputs, target_inputs, encoder_outputs):
    """p(target sentence | rest of the document), per example of the batch."""
    with torch.no_grad():
        # Exactly the original computation: no attention mask, and the padding of the target counts
        # towards both the summed loss and the number of tokens it is divided by.
        with torch.cuda.amp.autocast():
            outputs = model(encoder_outputs=encoder_outputs, labels=target_inputs['input_ids'],
                            return_dict=True)

        labels = target_inputs["input_ids"]

        logits = outputs.logits  # (batch_size, seq_length, vocab_size)

        # No manual shifting needed for BART!
        loss_per_token = F.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            labels.view(-1),
            ignore_index=-100,
            reduction="none"
        )

        loss_per_token = loss_per_token.view(labels.shape)

        valid_token_mask = (labels != -100).float()
        loss_per_example = loss_per_token.sum(dim=1) / valid_token_mask.sum(dim=1)

        conditional_probabilities_per_example = [math.exp(-l.item()) for l in loss_per_example]

    return conditional_probabilities_per_example


def generate_for_contexts(context_ids):
    """Generate for every context of the batch, running the beam search only once per distinct context.

    The context of a sentence is its document without that sentence, truncated to MAX_SOURCE_LENGTH tokens.
    Removing a sentence that begins after the truncation limit leaves the truncated context unchanged, so
    long documents produce several identical rows, and the generation only has to run for one of them. The
    rows are taken from the already padded batch unchanged, so each of them still holds exactly the tokens
    it holds in the original script.
    """
    rows = context_ids.tolist()

    first_row_of_context = {}
    unique_row_positions = []
    row_to_unique_position = []

    for row in rows:
        key = tuple(row)

        if key not in first_row_of_context:
            first_row_of_context[key] = len(unique_row_positions)
            unique_row_positions.append(len(row_to_unique_position))

        row_to_unique_position.append(first_row_of_context[key])

    unique_context_ids = context_ids[torch.tensor(unique_row_positions, device=context_ids.device)]

    generated_for_unique = model.generate(
        unique_context_ids,
        max_new_tokens=GENERATION_MAX_NEW_TOKENS,
        num_beams=GENERATION_NUM_BEAMS,
        no_repeat_ngram_size=GENERATION_NO_REPEAT_NGRAM_SIZE,
        early_stopping=True
    )

    return generated_for_unique[torch.tensor(row_to_unique_position, device=generated_for_unique.device)]


def marginal_probability(context_inputs, encoder_outputs):
    """p(the generation of the model | rest of the document), per example of the batch."""
    generated_output = generate_for_contexts(context_inputs['input_ids'])

    generated_output[generated_output == tokenizer.pad_token_id] = -100  # Mask padding tokens

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(encoder_outputs=encoder_outputs, labels=generated_output)

        logits = outputs.logits  # (batch_size, seq_length, vocab_size)

        # No shifting needed for BART!
        loss_per_token = F.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            generated_output.view(-1),  # Directly use generated labels
            ignore_index=-100,
            reduction="none"
        )

        loss_per_token = loss_per_token.view(generated_output.shape)

        valid_token_mask = (generated_output != -100).float()
        loss_per_example = loss_per_token.sum(dim=1) / valid_token_mask.sum(dim=1)

        marginal_probabilities_per_example = [math.exp(-l.item()) for l in loss_per_example]

    return marginal_probabilities_per_example


def calculate_pmi(target_sentences_per_text, docs_without_target_sentences_per_text):
    context_inputs = tokenizer(docs_without_target_sentences_per_text, return_tensors='pt', padding=True,
                               truncation=True, max_length=MAX_SOURCE_LENGTH).to(device)
    target_inputs = tokenizer(target_sentences_per_text, return_tensors='pt', padding=True, truncation=True,
                              max_length=MAX_SOURCE_LENGTH).to(device)

    encoder_outputs = encode_contexts(context_inputs)

    p_x_given_y_list = conditional_probability(context_inputs, target_inputs, encoder_outputs)
    p_x_list = marginal_probability(context_inputs, encoder_outputs)

    pmi_list = [math.log2(a / b) for a, b in zip(p_x_given_y_list, p_x_list)]

    return pmi_list


def single_process_calc_pmi_for_all(training_dataset):
    global all_selected_indices

    all_selected_indices = np.full(len(training_dataset), -1, dtype=np.int32)

    # Scores of the sentences of the examples that are not fully scored yet, in sentence order, and how
    # many sentences each of those examples has. Only the examples currently in flight are held, which is
    # at most one chunk worth of them.
    collected_scores = {}
    sentence_counts = {}

    # Sentences that did not fill a whole batch at the end of a chunk. They are carried over to the next
    # chunk instead of being sent as a smaller batch, so that the batches hold the same sentences as the
    # ones of the original script, which flattens the whole dataset into a single stream of sentences.
    pending = []

    def select_principal_sentence_of(example_pos):
        """Pick the sentence the original pipeline would pick, including how it breaks ties.

        The original keeps the top k sentences with np.argpartition, which returns them in partition order
        rather than in sentence order, and its combine script then takes the argmax over that order. When
        two sentences of a document score exactly the same - which happens when a document repeats a
        sentence, because both copies get the same context and therefore the same score - which of them
        wins depends on that order, so the same two steps are done here instead of tracking a running best.
        """
        scores = np.array(collected_scores.pop(example_pos))
        sentence_counts.pop(example_pos)

        # top k
        if len(scores) <= args.topk:
            ind = np.arange(len(scores))
        else:
            ind = np.argpartition(scores, -args.topk)[-args.topk:]

        # the argmax of the combine script, over the kept candidates in the order they were kept
        all_selected_indices[example_pos] = int(ind[int(np.argmax(scores[ind]))])

    def flush_pending(flush_all=False):
        """Score whole batches of the pending sentences and finish the examples that are complete."""
        while len(pending) >= SENTENCE_BATCH_SIZE or (flush_all and pending):
            batch = pending[:SENTENCE_BATCH_SIZE]
            del pending[:SENTENCE_BATCH_SIZE]

            pmi_scores = calculate_pmi([sentence for _, _, sentence, _ in batch],
                                       [context for _, _, _, context in batch])

            for (example_pos, _, _, _), score in zip(batch, pmi_scores):
                collected_scores[example_pos].append(score)

                if len(collected_scores[example_pos]) == sentence_counts[example_pos]:
                    select_principal_sentence_of(example_pos)

    # The documents are processed in chunks, so that only the sentences of DOC_CHUNK_SIZE documents
    # have to be kept in memory at any time.
    for chunk_start in tqdm.tqdm(range(0, len(training_dataset), DOC_CHUNK_SIZE), desc="Calculating PMI"):
        chunk_end = min(chunk_start + DOC_CHUNK_SIZE, len(training_dataset))

        chunk_texts = get_texts_of_chunk(training_dataset, chunk_start, chunk_end)

        sentences_per_doc = tokenization_pool.map(nltk.sent_tokenize, chunk_texts, chunksize=32)

        for doc_pos, (text, sentences) in enumerate(zip(chunk_texts, sentences_per_doc)):
            if not sentences:
                continue

            example_pos = chunk_start + doc_pos
            collected_scores[example_pos] = []
            sentence_counts[example_pos] = len(sentences)

            for sentence_index, sentence in enumerate(sentences):
                # The context is the document without the sentence, built the same way as in the original.
                pending.append((example_pos, sentence_index, sentence, text.replace(sentence, "", 1)))

        flush_pending()

    # The sentences of the last chunk are scored in a smaller final batch.
    flush_pending(flush_all=True)


def mask_out_principal_sentence(example, example_pos):
    """Write the final columns of the pretraining set: the masked document and its principal sentence.

    A new dictionary is returned instead of the modified example, because map() removes the columns of
    'remove_columns' from the example before it updates it with the returned columns. When the source set
    is a preprocessed one, its columns are named 'document' and 'summary' as well, so modifying the example
    in place would have both of them removed again.
    """
    selected = all_selected_indices[example_pos]

    if selected < 0:
        # The text of the example contains no sentence at all.
        return {"document": "", "summary": ""}

    sentences = nltk.sent_tokenize(get_text_of_example(example))

    return {
        "document": " ".join([s if j != selected else mask_token for j, s in enumerate(sentences)]),
        "summary": sentences[selected],
    }


if __name__ == "__main__":

    if args.source == "c4":
        dataset = load_dataset("c4", args.c4_split, cache_dir="./cache")
        dataset.pop("validation")
        source_columns = ["url", "text", "timestamp"]
    else:
        dataset = load_from_disk(args.source_dataset)
        source_columns = ["document", "summary"]

        print("\nSource documents are taken from: {}\n".format(args.source_dataset))

    if USE_SMALLER_SUBSET:
        ###  SUBSET_UPPER_LIMIT = len(dataset["train"])

        # If subset upper limit is larger than the dataset size, select until the end of the dataset.
        if SUBSET_UPPER_LIMIT > len(dataset["train"]):
            SUBSET_UPPER_LIMIT = len(dataset["train"])

        dataset["train"] = dataset["train"].select(list(range(SUBSET_LOWER_LIMIT, SUBSET_UPPER_LIMIT)))

    tokenization_pool = multiprocessing.Pool(TOKENIZE_NUM_PROC)

    single_process_calc_pmi_for_all(dataset["train"])

    tokenization_pool.close()
    tokenization_pool.join()

    # The model is not needed anymore and its memory is better spent on the map below.
    del model
    torch.cuda.empty_cache()

    # The mapped set is written to the cache directory first and is copied to OUTPUT_PATH afterwards,
    # so both of them have to fit on the disk. It is not kept in memory: at the size of the complete
    # realnewslike split that would need far more RAM than this machine has.
    os.makedirs(args.map_cache_dir, exist_ok=True)

    dataset["train"] = dataset["train"].map(
        mask_out_principal_sentence,
        with_indices=True,
        remove_columns=source_columns,
        batched=False,
        num_proc=MAP_NUM_PROC,
        keep_in_memory=False,
        cache_file_name=os.path.join(args.map_cache_dir, "pmi_fast_map_cache.arrow")
    )

    dataset.save_to_disk(OUTPUT_PATH)

    # The cache files hold a second copy of the whole set, which is no longer needed once it is saved.
    for cache_file in glob.glob(os.path.join(args.map_cache_dir, "pmi_fast_map_cache*.arrow")):
        os.remove(cache_file)

    print("\nThe preprocessed set was saved to: {}\n".format(OUTPUT_PATH))

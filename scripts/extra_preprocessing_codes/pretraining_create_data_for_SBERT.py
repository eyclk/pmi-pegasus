import argparse
import glob
import os
import multiprocessing
import nltk
from datasets import load_dataset, load_from_disk
import numpy as np
import torch
import tqdm
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel


SENTENCE_BATCH_SIZE = 256   # Number of sentences encoded by SBERT in one forward pass.
DOC_CHUNK_SIZE = 2000       # Number of documents processed (tokenized + encoded + scored) at a time.
MAX_SEQ_LENGTH = 256        # Same truncation limit that sentence-transformers uses for the all-* models.
USE_FP16 = True             # Half precision for the SBERT encoder (only used on GPU).
TOKENIZE_NUM_PROC = 16      # Worker processes that split the documents of a chunk into sentences.
MAP_NUM_PROC = 16           # Worker processes of the final map.

USE_SMALLER_SUBSET = False    # MODIFY HERE TO USE A SMALLER SUBSET OF THE DATASET
SUBSET_LOWER_LIMIT = 13000000
SUBSET_UPPER_LIMIT = 14000000


parser = argparse.ArgumentParser()

parser.add_argument("--c4_split", type=str, default="realnewslike", choices=["en", "realnewslike"])
parser.add_argument("--sbert_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
parser.add_argument("--doc_repr", type=str, default="mean_of_others", choices=["mean_of_others", "leave_one_out_text"])

# The source documents can be taken from an already preprocessed pretraining set instead of C4 itself:
# every 'document' of such a set is the source text with the principal sentence replaced by the mask token,
# so the source text is recovered by putting the 'summary' back in place of the mask. This avoids
# downloading C4 again and guarantees that this set and the source set are aligned example by example.
parser.add_argument("--source", type=str, default="preprocessed", choices=["c4", "preprocessed"])
parser.add_argument("--source_dataset", type=str,
                    default="./PREPROCESSED_DATASETS/c4_realnewslike_processed_ROUGE_complete_combined")

parser.add_argument("--map_cache_dir", type=str, default="./PREPROCESSED_DATASETS/sbert_map_cache",
                    help="Directory the map writes its cache files to. They are deleted once the set is saved.")

args = parser.parse_args()


# There is no separate combine step for SBERT, so the set this script writes is already the final one and is
# named accordingly. The ROUGE and PMI pipelines keep the top 'k' candidate sentences per example and pick the
# highest scoring one of them in their combine script, which - as long as the FactCC reranking stays disabled -
# is simply the highest scoring sentence of the whole example. That sentence is written directly here instead,
# which gives the exact same dataset while writing the documents only once rather than 'k' times.
OUTPUT_PATH = "./PREPROCESSED_DATASETS/c4_{}_processed_SBERT_complete_combined".format(args.c4_split)


mask_token = "<mask>"

# Load the SBERT tokenizer and encoder. sentence-transformers is deliberately NOT used here, because it would
# require upgrading 'transformers', which the rest of this codebase is pinned to. Mean pooling over the token
# embeddings followed by L2 normalization reproduces the sentence-transformers embeddings of the all-* models
# (their sentence-transformers config is Transformer -> Pooling(mean) -> Normalize).


def resolve_model_path(model_name_or_path):
    """Return a local directory that holds the model files.

    The old 'transformers' version of this environment cannot download from the current huggingface hub,
    so the files are fetched with 'huggingface_hub' first and are then loaded from the local snapshot.
    """
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    return snapshot_download(model_name_or_path, allow_patterns=["*.json", "*.txt", "*.bin", "*.model"])


model_path = resolve_model_path(args.sbert_model)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

if USE_FP16 and device.type == "cuda":
    model.half()

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


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(texts):
    """Encode a list of texts into L2-normalized SBERT embeddings (float32, on the current device)."""
    embeddings = []

    for i in range(0, len(texts), SENTENCE_BATCH_SIZE):
        batch = texts[i: i + SENTENCE_BATCH_SIZE]

        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                           max_length=MAX_SEQ_LENGTH).to(device)

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        batch_embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        embeddings.append(F.normalize(batch_embeddings.float(), p=2, dim=1))

    return torch.cat(embeddings, dim=0)


def calc_sbert_scores_for_one_doc(sentence_embeddings, sentences):
    """Score every sentence by the cosine similarity between the sentence and the rest of its document.

    This is the SBERT counterpart of the ROUGE-based selection of PEGASUS: instead of the lexical overlap
    between a sentence and the remaining document, the semantic similarity of their embeddings is used.
    """
    num_sentences = sentence_embeddings.shape[0]

    if num_sentences == 1:
        # A single sentence has no remaining document to be compared against.
        return np.zeros(1, dtype=np.float32)

    if args.doc_repr == "mean_of_others":
        # The remaining document is represented as the mean of the embeddings of all the other sentences.
        # This needs a single encoding pass per document and is not affected by the truncation limit of SBERT.
        summed = sentence_embeddings.sum(dim=0, keepdim=True)
        rest_embeddings = (summed - sentence_embeddings) / (num_sentences - 1)
    else:
        # The remaining document is encoded as an actual text, exactly like the ROUGE version builds it.
        # Long documents are truncated to MAX_SEQ_LENGTH tokens by SBERT.
        rest_texts = [" ".join([s for j, s in enumerate(sentences) if j != i]) for i in range(num_sentences)]
        rest_embeddings = encode_texts(rest_texts)

    rest_embeddings = F.normalize(rest_embeddings, p=2, dim=1)
    scores = (sentence_embeddings * rest_embeddings).sum(dim=1)

    return scores.cpu().numpy().astype(np.float32)


def single_process_calc_sbert_for_all(training_dataset):
    global all_selected_indices

    all_selected_indices = np.full(len(training_dataset), -1, dtype=np.int32)

    # The documents are processed in chunks, so that only the sentences of DOC_CHUNK_SIZE documents
    # have to be kept in memory at any time.
    for chunk_start in tqdm.tqdm(range(0, len(training_dataset), DOC_CHUNK_SIZE), desc="Calculating SBERT scores"):
        chunk_end = min(chunk_start + DOC_CHUNK_SIZE, len(training_dataset))

        chunk_texts = get_texts_of_chunk(training_dataset, chunk_start, chunk_end)

        sentences_per_doc = tokenization_pool.map(nltk.sent_tokenize, chunk_texts, chunksize=32)

        all_sentences = []
        for sentences in sentences_per_doc:
            all_sentences.extend(sentences)

        if len(all_sentences) == 0:
            continue

        all_sentence_embeddings = encode_texts(all_sentences)

        offset = 0
        for doc_pos, sentences in enumerate(sentences_per_doc):
            num_sentences = len(sentences)

            if num_sentences == 0:
                continue

            sentence_embeddings = all_sentence_embeddings[offset: offset + num_sentences]
            offset += num_sentences

            scores = calc_sbert_scores_for_one_doc(sentence_embeddings, sentences)

            # The principal sentence is the highest scoring sentence of the example.
            all_selected_indices[chunk_start + doc_pos] = int(np.argmax(scores))


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

    single_process_calc_sbert_for_all(dataset["train"])

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
        cache_file_name=os.path.join(args.map_cache_dir, "sbert_map_cache.arrow")
    )

    dataset.save_to_disk(OUTPUT_PATH)

    # The cache files hold a second copy of the whole set, which is no longer needed once it is saved.
    for cache_file in glob.glob(os.path.join(args.map_cache_dir, "sbert_map_cache*.arrow")):
        os.remove(cache_file)

    print("\nThe preprocessed set was saved to: {}\n".format(OUTPUT_PATH))

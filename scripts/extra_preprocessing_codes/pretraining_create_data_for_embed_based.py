"""Principal sentence selection by a plain embedding based semantic method.

The selection criterion is the same one pretraining_create_data_for_SBERT.py uses: every sentence is
embedded, the rest of its document is represented by the mean of the embeddings of the other sentences,
and the sentence with the highest cosine similarity to that is selected. What differs is the encoder.

The SBERT script uses a sentence-transformers model, that is an encoder that was trained on sentence pairs
specifically so that cosine similarity between its embeddings is meaningful. This script uses an ordinary
pre-trained language model whose token embeddings are simply averaged, with no such training - the
'Avg. BERT embeddings' baseline of the sentence-BERT paper (Reimers and Gurevych, 2019), which that paper
introduced sentence-BERT to improve upon. The two sets therefore differ only in the quality of the
embeddings, and not in how a sentence is selected from them.

--embedding_model takes any model whose averaged token embeddings should be used, so the same criterion can
also be run with, for example, bert-large-uncased or roberta-base. The name of the model is part of the
name of the set that is written, so runs with different encoders cannot overwrite each other.
"""

import argparse
import glob
import os

# The large model files are served through the Xet CDN ('us.aws.cdn.hf.co') by default, which is not
# reachable from every machine while 'huggingface.co' itself is. Falling back to the ordinary CDN keeps
# the download working there. It is set before huggingface_hub is imported, because that is when the
# setting is read, and only if it was not set explicitly in the environment already.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import multiprocessing
import fsspec
import nltk
import datasets.arrow_dataset
import datasets.dataset_dict
from datasets import load_dataset, load_from_disk
import numpy as np
import torch
import tqdm
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel


def apply_local_filesystem_compatibility_patch():
    """Stop old 'datasets' versions from copying local datasets through a temporary directory.

    datasets 2.0 decides whether a path is remote with 'fs.protocol != "file"', while fsspec has been
    reporting ('file', 'local') for the local filesystem since 2023. The local disk is therefore taken for
    a remote one, and load_from_disk and save_to_disk copy the whole dataset into a temporary directory
    first - which for a set of this size means tens of GB into /tmp, and usually 'No space left on device'.
    """
    if not hasattr(datasets.arrow_dataset, "is_remote_filesystem"):
        return  # Newer datasets versions do not have this problem.

    if not datasets.arrow_dataset.is_remote_filesystem(fsspec.filesystem("file")):
        return  # The installed combination of versions detects the local filesystem correctly.

    def is_remote_filesystem(fs):
        if fs is None:
            return False

        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
        return protocol != "file"

    datasets.arrow_dataset.is_remote_filesystem = is_remote_filesystem
    datasets.dataset_dict.is_remote_filesystem = is_remote_filesystem


apply_local_filesystem_compatibility_patch()


SENTENCE_BATCH_SIZE = 256   # Number of sentences encoded in one forward pass.
DOC_CHUNK_SIZE = 2000       # Number of documents processed (tokenized + encoded + scored) at a time.
MAX_SEQ_LENGTH = 256        # Enough for a sentence, and the limit the SBERT set was built with.
USE_FP16 = True             # Half precision for the encoder (only used on GPU).
TOKENIZE_NUM_PROC = 16      # Worker processes that split the documents of a chunk into sentences.
MAP_NUM_PROC = 16           # Worker processes of the final map.

USE_SMALLER_SUBSET = False    # MODIFY HERE TO USE A SMALLER SUBSET OF THE DATASET
SUBSET_LOWER_LIMIT = 13000000
SUBSET_UPPER_LIMIT = 14000000


parser = argparse.ArgumentParser()

parser.add_argument("--c4_split", type=str, default="realnewslike", choices=["en", "realnewslike"])
parser.add_argument("--embedding_model", type=str, default="bert-base-uncased")

# The source documents can be taken from an already preprocessed pretraining set instead of C4 itself:
# every 'document' of such a set is the source text with the principal sentence replaced by the mask token,
# so the source text is recovered by putting the 'summary' back in place of the mask. This avoids
# downloading C4 again and guarantees that this set and the source set are aligned example by example.
# Only the sentences are embedded here, and those are the same either way.
parser.add_argument("--source", type=str, default="preprocessed", choices=["c4", "preprocessed"])
parser.add_argument("--source_dataset", type=str,
                    default="./PREPROCESSED_DATASETS/c4_realnewslike_processed_ROUGE_complete_combined")

parser.add_argument("--map_cache_dir", type=str, default="./PREPROCESSED_DATASETS/embed_based_map_cache",
                    help="Directory the map writes its cache files to. They are deleted once the set is saved.")

args = parser.parse_args()


# The encoder is part of the name, so that runs with different encoders cannot overwrite each other. There
# is no separate combine step: the combine script of the ROUGE and PMI pipelines picks the highest scoring
# of the top k candidate sentences, which is simply the highest scoring sentence of the example, and that
# one is written directly here, which gives the same dataset while writing the documents only once.
OUTPUT_PATH = "./PREPROCESSED_DATASETS/c4_{}_processed_EMBED_{}_complete_combined".format(
    args.c4_split, os.path.basename(args.embedding_model.rstrip("/")))


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
        return snapshot_download(
            model_name_or_path,
            allow_patterns=["*.json", "*.txt", "*.bin", "*.model"],
            # Some repositories also hold quantized and openvino copies of the weights, which are not used
            # here and only slow the download down.
            ignore_patterns=["openvino/*", "onnx/*", "*qint8*", "*quantized*"],
        )
    except Exception as download_error:
        raise RuntimeError(
            "The embedding model '{}' could not be downloaded from the huggingface hub: {}\n\n"
            "If this machine cannot reach the hub, download the model on a machine that can and pass the "
            "directory it was saved in with --embedding_model, for example:\n"
            "    huggingface-cli download {} --local-dir ./embedding_model\n"
            "    python {} --embedding_model ./embedding_model".format(
                model_name_or_path, download_error, model_name_or_path, os.path.basename(__file__))
        ) from download_error


model_path = resolve_model_path(args.embedding_model)

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


def encode_sentences(sentences):
    """Embed sentences as the average of their token embeddings, L2-normalized for cosine similarity.

    The sentences are batched by length rather than in the order they appear in. A batch is padded to its
    longest sentence, and sentences of a document range from a few tokens to the truncation limit, so
    batching them as they come spends most of the computation on padding: measured on real realnewslike
    sentences, 3.45 times as many padded tokens as real ones, against 1.05 times when they are sorted.
    This does not change the embeddings, since the attention mask keeps the padding out of the attention
    and the mean pooling divides by the number of real tokens - only the speed changes, by about 3 times.
    """
    # Tokenize once without padding, so that the lengths are known and the batches can be padded per batch.
    encoded = tokenizer(sentences, truncation=True, max_length=MAX_SEQ_LENGTH)

    order = sorted(range(len(sentences)), key=lambda i: len(encoded["input_ids"][i]))

    embeddings = torch.empty((len(sentences), model.config.hidden_size), dtype=torch.float32, device=device)

    for i in range(0, len(order), SENTENCE_BATCH_SIZE):
        batch_indices = order[i: i + SENTENCE_BATCH_SIZE]

        inputs = tokenizer.pad({key: [values[j] for j in batch_indices] for key, values in encoded.items()},
                               return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        batch_embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])

        # Put the embeddings back where their sentences are.
        embeddings[torch.tensor(batch_indices, device=device)] = F.normalize(batch_embeddings.float(), p=2, dim=1)

    return embeddings


def calc_similarity_to_rest_of_document(sentence_embeddings):
    """Cosine similarity between every sentence and the mean of the embeddings of the other sentences."""
    num_sentences = sentence_embeddings.shape[0]

    if num_sentences == 1:
        # A single sentence has no remaining document to be compared against.
        return torch.zeros(1, device=sentence_embeddings.device)

    summed = sentence_embeddings.sum(dim=0, keepdim=True)
    rest_embeddings = F.normalize((summed - sentence_embeddings) / (num_sentences - 1), p=2, dim=1)

    return (sentence_embeddings * rest_embeddings).sum(dim=1)


def calc_scores_and_select_for_all(training_dataset):
    global all_selected_indices

    all_selected_indices = np.full(len(training_dataset), -1, dtype=np.int32)

    # The documents are processed in chunks, so that only the sentences of DOC_CHUNK_SIZE documents
    # have to be kept in memory at any time.
    for chunk_start in tqdm.tqdm(range(0, len(training_dataset), DOC_CHUNK_SIZE),
                                 desc="Selecting principal sentences"):
        chunk_end = min(chunk_start + DOC_CHUNK_SIZE, len(training_dataset))

        chunk_texts = get_texts_of_chunk(training_dataset, chunk_start, chunk_end)

        sentences_per_doc = tokenization_pool.map(nltk.sent_tokenize, chunk_texts, chunksize=32)

        all_sentences = []
        for sentences in sentences_per_doc:
            all_sentences.extend(sentences)

        if not all_sentences:
            continue

        all_sentence_embeddings = encode_sentences(all_sentences)

        offset = 0
        for doc_pos, sentences in enumerate(sentences_per_doc):
            num_sentences = len(sentences)

            if num_sentences == 0:
                continue

            embeddings_of_doc = all_sentence_embeddings[offset: offset + num_sentences]
            offset += num_sentences

            scores = calc_similarity_to_rest_of_document(embeddings_of_doc)

            all_selected_indices[chunk_start + doc_pos] = int(torch.argmax(scores).item())


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

    calc_scores_and_select_for_all(dataset["train"])

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
        cache_file_name=os.path.join(args.map_cache_dir, "embed_based_map_cache.arrow")
    )

    dataset.save_to_disk(OUTPUT_PATH)

    # The cache files hold a second copy of the whole set, which is no longer needed once it is saved.
    for cache_file in glob.glob(os.path.join(args.map_cache_dir, "embed_based_map_cache*.arrow")):
        os.remove(cache_file)

    print("\nThe preprocessed set was saved to: {}\n".format(OUTPUT_PATH))

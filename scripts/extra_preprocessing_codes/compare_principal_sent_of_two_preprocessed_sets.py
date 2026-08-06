"""Compare the principal sentences selected by two different preprocessing methods.

Both inputs must be pretraining sets produced by a 'pretraining_create_data_for_*.py' script followed by
its 'pretraining_combine_scores_for_*.py' script, i.e. datasets with a 'document' and a 'summary' column,
where 'summary' is the selected principal sentence. The two sets must be built from the same slice of C4,
so that the examples at the same index correspond to the same source document.

Example:
    python compare_principal_sent_of_two_preprocessed_sets.py \
        --dataset_a ./PREPROCESSED_DATASETS/c4_realnewslike_processed_ROUGE_complete_combined \
        --dataset_b ./PREPROCESSED_DATASETS/c4_realnewslike_processed_SBERT_complete_combined \
        --name_a ROUGE --name_b SBERT
"""

import argparse
import json
import fsspec
import datasets.arrow_dataset
import datasets.dataset_dict
from datasets import load_from_disk, DatasetDict
from tqdm import tqdm


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



CHUNK_SIZE = 10000  # Number of examples fetched from each dataset at a time.

mask_token = "<mask>"


parser = argparse.ArgumentParser()

parser.add_argument("--dataset_a", type=str, required=True)
parser.add_argument("--dataset_b", type=str, required=True)
parser.add_argument("--name_a", type=str, default="DATASET A")
parser.add_argument("--name_b", type=str, default="DATASET B")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--max_examples", type=int, default=-1, help="Compare only the first N examples. -1 uses all of them.")
parser.add_argument("--check_alignment", action="store_true",
                    help="Verify that the compared examples originate from the same source document.")
parser.add_argument("--print_first_k_differences", type=int, default=0)
parser.add_argument("--save_differences_to", type=str, default="",
                    help="Optional path of a json file the differing examples are written to.")

args = parser.parse_args()


def load_split(dataset_path, split):
    dataset = load_from_disk(dataset_path)

    if isinstance(dataset, DatasetDict):
        assert split in dataset, "Split '{}' is not present in {}. Available splits: {}".format(
            split, dataset_path, list(dataset.keys()))
        dataset = dataset[split]

    assert "summary" in dataset.column_names, \
        "'{}' must have a 'summary' column that holds the principal sentence.".format(dataset_path)
    assert "document" in dataset.column_names, \
        "'{}' must have a 'document' column.".format(dataset_path)

    return dataset


def normalize_sentence(sentence):
    # The two preprocessing runs may keep slightly different unicode variants of the same characters,
    # so these are unified before the sentences are compared.
    return (sentence.replace("’", "'").replace("‘", "'").replace("“", "\"")
            .replace("”", "\"").replace("–", "-").replace("­", "-").strip())


def compare_principal_sentences():
    dataset_a = load_split(args.dataset_a, args.split)
    dataset_b = load_split(args.dataset_b, args.split)

    assert len(dataset_a) == len(dataset_b), \
        "Both datasets must have the same number of examples ({} vs {}).".format(len(dataset_a), len(dataset_b))

    total_examples = len(dataset_a)
    if args.max_examples > 0:
        total_examples = min(total_examples, args.max_examples)

    same_principal_sentence_count = 0
    different_principal_sentence_count = 0
    almost_same_with_minor_differences_count = 0
    misaligned_example_count = 0

    differences = []

    for chunk_start in tqdm(range(0, total_examples, CHUNK_SIZE), desc="Comparing principal sentences"):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_examples)

        chunk_a = dataset_a[chunk_start: chunk_end]
        chunk_b = dataset_b[chunk_start: chunk_end]

        for pos in range(chunk_end - chunk_start):
            summary_a = normalize_sentence(chunk_a["summary"][pos])
            summary_b = normalize_sentence(chunk_b["summary"][pos])

            if args.check_alignment:
                # The document is the source text with the principal sentence replaced by the mask token,
                # so putting the principal sentence back has to give the same source text for both datasets.
                source_a = normalize_sentence(chunk_a["document"][pos].replace(mask_token, chunk_a["summary"][pos]))
                source_b = normalize_sentence(chunk_b["document"][pos].replace(mask_token, chunk_b["summary"][pos]))

                if source_a != source_b:
                    misaligned_example_count += 1

            if summary_a == summary_b:
                same_principal_sentence_count += 1
                continue

            if len(summary_a) == len(summary_b) and summary_a[:20] == summary_b[:20]:
                # Same sentence apart from a few characters that the normalization above does not cover.
                almost_same_with_minor_differences_count += 1
            else:
                different_principal_sentence_count += 1

            if args.save_differences_to or len(differences) < args.print_first_k_differences:
                differences.append({
                    "index": chunk_start + pos,
                    args.name_a: chunk_a["summary"][pos],
                    args.name_b: chunk_b["summary"][pos],
                })

    # The examples with minor differences are also examples of different principal sentences.
    different_principal_sentence_count += almost_same_with_minor_differences_count

    print("\nComparison of the principal sentences of '{}' and '{}':".format(args.name_a, args.name_b))
    print("\n{}: {}".format(args.name_a, args.dataset_a))
    print("{}: {}".format(args.name_b, args.dataset_b))

    print("\nTotal examples compared: {}".format(total_examples))
    print("\nNumber of examples with the same principal sentence: {} ({:.2f}%)".format(
        same_principal_sentence_count, 100 * same_principal_sentence_count / max(total_examples, 1)))
    print("Number of examples with different principal sentences: {} ({:.2f}%)".format(
        different_principal_sentence_count, 100 * different_principal_sentence_count / max(total_examples, 1)))
    print("Number of examples with almost the same principal sentences but minor differences: {}\n".format(
        almost_same_with_minor_differences_count))

    if args.check_alignment:
        if misaligned_example_count == 0:
            print("Alignment check passed: every compared pair comes from the same source document.\n")
        else:
            print("===> WARNING: {} example pairs do NOT come from the same source document. "
                  "The two datasets are probably built from different slices of C4, "
                  "which makes the comparison above meaningless.\n".format(misaligned_example_count))

    for difference in differences[:args.print_first_k_differences]:
        print("\n----- Example {}:".format(difference["index"]))
        print("*** {} principal sentence:\n{}\n".format(args.name_a, difference[args.name_a]))
        print("*** {} principal sentence:\n{}".format(args.name_b, difference[args.name_b]))

    if args.save_differences_to:
        with open(args.save_differences_to, "w", encoding="utf-8") as output_file:
            json.dump(differences, output_file, ensure_ascii=False, indent=2)
        print("\nThe {} differing examples were saved to: {}\n".format(len(differences), args.save_differences_to))


if __name__ == "__main__":
    compare_principal_sentences()

"""Compare the principal sentences selected by three different preprocessing methods.

All three inputs must be pretraining sets produced by a 'pretraining_create_data_for_*.py' script followed by
its 'pretraining_combine_scores_for_*.py' script, i.e. datasets with a 'document' and a 'summary' column,
where 'summary' is the selected principal sentence. The three sets must be built from the same slice of C4,
so that the examples at the same index correspond to the same source document.

The first two sets are always the PMI and the ROUGE ones, the third one is whichever set they are compared
against, currently the EMBED (bert-base-uncased) or the SBERT set.

Example:
    python compare_principal_sent_of_three_preprocessed_sets.py \
        --dataset_pmi ./PREPROCESSED_DATASETS/c4_realnewslike_processed_PMI_complete_combined \
        --dataset_rouge ./PREPROCESSED_DATASETS/c4_realnewslike_processed_ROUGE_complete_combined \
        --dataset_other ./PREPROCESSED_DATASETS/c4_realnewslike_processed_SBERT_complete_combined \
        --name_other SBERT
        --log_to ./diff_PMI_vs_ROUGE_vs_SBERT__output.txt
"""

import argparse
import os
import sys
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

script_directory = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser()

parser.add_argument("--dataset_pmi", type=str, required=True)
parser.add_argument("--dataset_rouge", type=str, required=True)
parser.add_argument("--dataset_other", type=str, required=True,
                    help="The third set the PMI and the ROUGE ones are compared against.")
parser.add_argument("--name_pmi", type=str, default="PMI")
parser.add_argument("--name_rouge", type=str, default="ROUGE")
parser.add_argument("--name_other", type=str, default="EMBED",
                    help="Name of the third set, used in the output and in the name of the log file.")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--max_examples", type=int, default=-1, help="Compare only the first N examples. -1 uses all of them.")
parser.add_argument("--no_check_alignment", dest="check_alignment", action="store_false",
                    help="Skip the verification that the compared examples originate from the same source document.")
parser.add_argument("--print_first_k_differences", type=int, default=0,
                    help="Print the first K examples on which the three sets do not fully agree.")
parser.add_argument("--log_to", type=str, default="",
                    help="Path of the log file the output is written to. Empty means a "
                         "'diff_<name_pmi>_vs_<name_rouge>_vs_<name_other>__output.txt' file "
                         "next to this script.")

parser.set_defaults(check_alignment=True)

args = parser.parse_args()

if not args.log_to:
    args.log_to = os.path.join(script_directory, "diff_{}_vs_{}_vs_{}__output.txt".format(
        args.name_pmi, args.name_rouge, args.name_other))


log_lines = []


def log(line=""):
    """Print a line and keep it for the log file."""
    print(line)
    log_lines.append(line)


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
    # The three preprocessing runs may keep slightly different unicode variants of the same characters,
    # so these are unified before the sentences are compared.
    return (sentence.replace("’", "'").replace("‘", "'").replace("“", "\"")
            .replace("”", "\"").replace("–", "-").replace("­", "-").strip())


def is_almost_the_same(first_sentence, second_sentence):
    # Same sentence apart from a few characters that the normalization above does not cover.
    return len(first_sentence) == len(second_sentence) and first_sentence[:20] == second_sentence[:20]


def percentage_of(count, total):
    return 100 * count / max(total, 1)


def compare_principal_sentences():
    names = [args.name_pmi, args.name_rouge, args.name_other]
    paths = [args.dataset_pmi, args.dataset_rouge, args.dataset_other]

    datasets_to_compare = [load_split(path, args.split) for path in paths]

    lengths = [len(dataset) for dataset in datasets_to_compare]
    assert len(set(lengths)) == 1, \
        "All three datasets must have the same number of examples ({}).".format(
            ", ".join("{}: {}".format(name, length) for name, length in zip(names, lengths)))

    total_examples = lengths[0]
    if args.max_examples > 0:
        total_examples = min(total_examples, args.max_examples)

    # The five mutually exclusive outcomes of comparing three principal sentences.
    all_three_same_count = 0
    only_pmi_and_rouge_same_count = 0    # The third set picked another sentence.
    only_pmi_and_other_same_count = 0    # ROUGE picked another sentence.
    only_rouge_and_other_same_count = 0  # PMI picked another sentence.
    all_three_different_count = 0

    # Informational: pairs that differ only by characters the normalization does not cover.
    almost_same_pair_counts = [0, 0, 0]  # PMI/ROUGE, PMI/other, ROUGE/other.

    misaligned_example_count = 0

    differences = []

    for chunk_start in tqdm(range(0, total_examples, CHUNK_SIZE), desc="Comparing principal sentences"):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_examples)

        chunks = [dataset[chunk_start: chunk_end] for dataset in datasets_to_compare]

        for pos in range(chunk_end - chunk_start):
            summaries = [chunk["summary"][pos] for chunk in chunks]
            pmi_summary, rouge_summary, other_summary = [normalize_sentence(summary) for summary in summaries]

            if args.check_alignment:
                # The document is the source text with the principal sentence replaced by the mask token,
                # so putting the principal sentence back has to give the same source text for all three sets.
                sources = [normalize_sentence(chunk["document"][pos].replace(mask_token, chunk["summary"][pos]))
                           for chunk in chunks]

                if len(set(sources)) != 1:
                    misaligned_example_count += 1

            pmi_equals_rouge = pmi_summary == rouge_summary
            pmi_equals_other = pmi_summary == other_summary
            rouge_equals_other = rouge_summary == other_summary

            if pmi_equals_rouge and pmi_equals_other:
                all_three_same_count += 1
                continue

            if pmi_equals_rouge:
                only_pmi_and_rouge_same_count += 1
            elif pmi_equals_other:
                only_pmi_and_other_same_count += 1
            elif rouge_equals_other:
                only_rouge_and_other_same_count += 1
            else:
                all_three_different_count += 1

            if not pmi_equals_rouge and is_almost_the_same(pmi_summary, rouge_summary):
                almost_same_pair_counts[0] += 1
            if not pmi_equals_other and is_almost_the_same(pmi_summary, other_summary):
                almost_same_pair_counts[1] += 1
            if not rouge_equals_other and is_almost_the_same(rouge_summary, other_summary):
                almost_same_pair_counts[2] += 1

            if len(differences) < args.print_first_k_differences:
                differences.append({
                    "index": chunk_start + pos,
                    "summaries": summaries,
                })

    # Overall pairwise agreement, i.e. the examples on which the two sets picked the same sentence,
    # whether or not the third set agreed as well.
    pmi_and_rouge_agree_count = all_three_same_count + only_pmi_and_rouge_same_count
    pmi_and_other_agree_count = all_three_same_count + only_pmi_and_other_same_count
    rouge_and_other_agree_count = all_three_same_count + only_rouge_and_other_same_count

    log("\nComparison of the principal sentences of '{}', '{}' and '{}':".format(*names))
    log()
    for name, path in zip(names, paths):
        log("{}: {}".format(name, path))

    log("\nTotal examples compared: {}".format(total_examples))

    log("\n--- Agreement of all three sets ---\n")
    log("Same principal sentence in all three sets: {} ({:.2f}%)".format(
        all_three_same_count, percentage_of(all_three_same_count, total_examples)))
    log("Completely different principal sentence in all three sets: {} ({:.2f}%)".format(
        all_three_different_count, percentage_of(all_three_different_count, total_examples)))

    log("\n--- Exactly two sets agree, the third one picked another sentence ---\n")
    log("{} = {}, but {} differs: {} ({:.2f}%)".format(
        names[0], names[1], names[2],
        only_pmi_and_rouge_same_count, percentage_of(only_pmi_and_rouge_same_count, total_examples)))
    log("{} = {}, but {} differs: {} ({:.2f}%)".format(
        names[0], names[2], names[1],
        only_pmi_and_other_same_count, percentage_of(only_pmi_and_other_same_count, total_examples)))
    log("{} = {}, but {} differs: {} ({:.2f}%)".format(
        names[1], names[2], names[0],
        only_rouge_and_other_same_count, percentage_of(only_rouge_and_other_same_count, total_examples)))

    log("\n--- Pairwise agreement, regardless of the third set ---\n")
    log("{} = {}: {} ({:.2f}%)".format(
        names[0], names[1], pmi_and_rouge_agree_count, percentage_of(pmi_and_rouge_agree_count, total_examples)))
    log("{} = {}: {} ({:.2f}%)".format(
        names[0], names[2], pmi_and_other_agree_count, percentage_of(pmi_and_other_agree_count, total_examples)))
    log("{} = {}: {} ({:.2f}%)".format(
        names[1], names[2], rouge_and_other_agree_count, percentage_of(rouge_and_other_agree_count, total_examples)))

    log("\n--- Pairs that are almost the same but have minor differences ---\n")
    log("These are counted as different sentences above.\n")
    log("{} and {}: {}".format(names[0], names[1], almost_same_pair_counts[0]))
    log("{} and {}: {}".format(names[0], names[2], almost_same_pair_counts[1]))
    log("{} and {}: {}".format(names[1], names[2], almost_same_pair_counts[2]))

    if args.check_alignment:
        if misaligned_example_count == 0:
            log("\nAlignment check passed: every compared triple comes from the same source document.")
        else:
            log("\n===> WARNING: {} example triples do NOT come from the same source document. "
                "The three datasets are probably built from different slices of C4, "
                "which makes the comparison above meaningless.".format(misaligned_example_count))

    for difference in differences[:args.print_first_k_differences]:
        log("\n\n----- Example {}:".format(difference["index"]))
        for name, summary in zip(names, difference["summaries"]):
            log("\n*** {} principal sentence:\n{}".format(name, summary))

    with open(args.log_to, "w", encoding="utf-8") as log_file:
        log_file.write("Command: {}\n".format(" ".join(sys.argv)))
        log_file.write("\n".join(log_lines) + "\n")

    print("\nThe output above was saved to: {}\n".format(args.log_to))


if __name__ == "__main__":
    compare_principal_sentences()

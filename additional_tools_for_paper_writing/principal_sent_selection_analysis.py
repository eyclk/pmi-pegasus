import datasets
# import json
import pandas as pd
from tqdm import tqdm


all_datasets_folder_path = "D:\\PMI-Pegasus Project - EXTRAS\\PREPROCESSED_DATASETS_BACKUPS - 26.08.2025\\"

simplified_datasets_path = "D:\\PMI-Pegasus Project - EXTRAS\\PREPROCESSED_DATASETS_BACKUPS - 26.08.2025\\SIMPLIFIED_DATASETS\\"


#####     Possible dataset names:
# dataset_for_PMI_pegasus_1_MIL
# dataset_for_PMI_pegasus_Complete
# dataset_for_ROUGE_pegasus_1_MIL
# dataset_for_ROUGE_pegasus_Complete


def analyze_principal_sent_selection(dataset_name="dataset_for_PMI_pegasus_1_MIL", split="train", do_select_first_k=True, first_k_examples=10, do_save_to_json=False):
    # combine the folder path and dataset name to get the full path
    dataset_path = all_datasets_folder_path + dataset_name

    # load the dataset
    dataset = datasets.load_from_disk(dataset_path)[split]

    # If do_select_first is True, select only the first_k_examples examples
    if do_select_first_k:
        dataset = dataset.select(range(first_k_examples))

    total_examples = len(dataset)
    print(f"\nTotal examples in the pretrain set for \'{dataset_name}\': {total_examples}\n")

    # Get the name of all columns
    column_names = dataset.column_names
    print(f"\nColumn names in the dataset: {column_names}\n")

    if do_select_first_k:
        counter = 0
        # Iterate through the dataset and print the first 10 examples
        for example in dataset:
            print(f"\n----- Example {counter+1}/{first_k_examples}:\n")
            print(f"Document:\n{example['document']}\n")
            print(f"Principal Sentence:\n{example['summary']}\n")
            print("-----\n")
            counter += 1

    if do_save_to_json:
        # Iterate through the dataset and place into a pandas DataFrame
        df = pd.DataFrame(dataset)
        # Rename the columns into "document" and "principal_sentence"
        df = df.rename(columns={"document": "document", "summary": "principal_sentence"})

        # Write the df object in json format to a file in this path: simplified_datasets_path + dataset_name + "__simplified.json"
        output_file_path = simplified_datasets_path + dataset_name + "__simplified.json"
        df.to_json(output_file_path, orient="records", lines=True)
        print(f"\nSimplified dataset saved to: {output_file_path}\n")


def compare_both_datasets_for_different_principal_sentences():
    # Load both complete datasets for PMI and ROUGE
    dataset_pmi_complete = datasets.load_from_disk(all_datasets_folder_path + "dataset_for_PMI_pegasus_Complete")["train"]
    dataset_rouge_complete = datasets.load_from_disk(all_datasets_folder_path + "dataset_for_ROUGE_pegasus_Complete")["train"]

    # Check that both datasets have the same number of examples using assert
    assert len(dataset_pmi_complete) == len(dataset_rouge_complete), "Both datasets must have the same number of examples."

    # Check if both datasets have the same column names using assert.
    # 'document' is the column name for the document in both datasets.
    assert "document" in dataset_pmi_complete.column_names, "The PMI dataset must have a 'document' column for the document."
    assert "document" in dataset_rouge_complete.column_names, "The ROUGE dataset must have a 'document' column for the document."
    # 'summary' is the column name for the principal sentence in both datasets.
    assert "summary" in dataset_pmi_complete.column_names, "The PMI dataset must have a 'summary' column for the principal sentence."
    assert "summary" in dataset_rouge_complete.column_names, "The ROUGE dataset must have a 'summary' column for the principal sentence."

    # Iterate through both datasets and compare the principal sentences for each example.
    # Count how many examples have the same principal sentence and how many have different principal sentences.
    same_principal_sentence_count = 0
    different_principal_sentence_count = 0
    almost_same_with_minor_differences_count = 0

    for example_pmi, example_rouge in tqdm(zip(dataset_pmi_complete, dataset_rouge_complete), total=len(dataset_pmi_complete), desc="Comparing principal sentences"):
        # Replace ’ with ' in both principal sentences before comparing, and also strip leading and trailing whitespace.
        example_pmi["summary"] = (example_pmi["summary"].replace("’", "'").replace("‘", "'").replace("“", "\"")
                                  .replace("”", "\"").replace("–", "-").replace("­", "-").strip())
        example_rouge["summary"] = (example_rouge["summary"].replace("’", "'").replace("‘", "'").replace("“", "\"")
                                    .replace("”", "\"").replace("–", "-").replace("­", "-").strip())

        if example_pmi["summary"].strip() == example_rouge["summary"].strip():
            same_principal_sentence_count += 1
        elif len(example_pmi["summary"].strip()) == len(example_rouge["summary"].strip()) and example_pmi["summary"].strip()[:20] == example_rouge["summary"].strip()[:20]:
            # print("========>> ERROR: Both principal sentences have the same length (and their first few characters are the same) but are different. This should not happen. Please check the datasets for this example.")
            # print(f"Document:\n{example_pmi['document']}\n")
            # print(f"*** PMI Principal Sentence:\n{example_pmi['summary']}\n")
            # print(f"*** ROUGE Principal Sentence:\n{example_rouge['summary']}\n")
            almost_same_with_minor_differences_count += 1
        else:
            different_principal_sentence_count += 1

    # Add almost_same_with_minor_differences_count to the different_principal_sentence_count, since they are also examples of different principal sentences, just with minor differences.
    different_principal_sentence_count += almost_same_with_minor_differences_count

    print("\nComparison of Principal Sentences between PMI and ROUGE Complete Datasets:")
    print(f"\nTotal examples compared: {len(dataset_pmi_complete)}")
    print(f"\nNumber of examples with the same principal sentence: {same_principal_sentence_count}")
    print(f"\nNumber of examples with different principal sentences: {different_principal_sentence_count}")
    print(f"\nNumber of examples with almost the same principal sentences but minor differences: {almost_same_with_minor_differences_count}\n")



# analyze_principal_sent_selection(dataset_name="dataset_for_PMI_pegasus_1_MIL", do_select_first_k=True, first_k_examples=20)
# analyze_principal_sent_selection(dataset_name="dataset_for_PMI_pegasus_1_MIL", do_save_to_json=True)

# analyze_principal_sent_selection(dataset_name="dataset_for_PMI_pegasus_Complete", do_select_first_k=True, first_k_examples=20)
# analyze_principal_sent_selection(dataset_name="dataset_for_PMI_pegasus_Complete", do_save_to_json=True)

# analyze_principal_sent_selection(dataset_name="dataset_for_ROUGE_pegasus_1_MIL", do_select_first_k=True, first_k_examples=20)
# analyze_principal_sent_selection(dataset_name="dataset_for_ROUGE_pegasus_1_MIL", do_save_to_json=True)

# analyze_principal_sent_selection(dataset_name="dataset_for_ROUGE_pegasus_Complete", do_select_first_k=True, first_k_examples=20)
# analyze_principal_sent_selection(dataset_name="dataset_for_ROUGE_pegasus_Complete", do_save_to_json=True)

compare_both_datasets_for_different_principal_sentences()

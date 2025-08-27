import datasets
import json
import pandas as pd


all_datasets_folder_path = "C:\\Users\\yigit\\Desktop\\PMI-Pegasus Project\\Step 4 - More analyses & Starting our paper\\4- Analysis of principal sentences\\PREPROCESSED_DATASETS_BACKUPS\\"

simplified_datasets_path = "C:\\Users\\yigit\\Desktop\\PMI-Pegasus Project\\Step 4 - More analyses & Starting our paper\\4- Analysis of principal sentences\\Simplified_datasets\\"


#####     Possible dataset names:
# dataset_for_PMI_pegasus_1_MIL
# dataset_for_PMI_pegasus_Complete
# dataset_for_ROUGE_pegasus_1_MIL
# dataset_for_ROUGE_pegasus_Complete


def analyze_principal_sent_selection(dataset_name="dataset_for_PMI_pegasus_1_MIL", split="train", do_select_first_k=False, first_k_examples=10, do_save_to_json=False):
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


# analyze_principal_sent_selection(dataset_name="dataset_for_PMI_pegasus_1_MIL", do_select_first_k=True, first_k_examples=50)
# analyze_principal_sent_selection(dataset_name="dataset_for_PMI_pegasus_1_MIL", do_save_to_json=True)

# analyze_principal_sent_selection(dataset_name="dataset_for_PMI_pegasus_Complete", do_select_first_k=True, first_k_examples=50)
# analyze_principal_sent_selection(dataset_name="dataset_for_PMI_pegasus_Complete", do_save_to_json=True)

# analyze_principal_sent_selection(dataset_name="dataset_for_ROUGE_pegasus_1_MIL", do_select_first_k=True, first_k_examples=50)
# analyze_principal_sent_selection(dataset_name="dataset_for_ROUGE_pegasus_1_MIL", do_save_to_json=True)

analyze_principal_sent_selection(dataset_name="dataset_for_ROUGE_pegasus_Complete", do_select_first_k=True, first_k_examples=50)
# analyze_principal_sent_selection(dataset_name="dataset_for_ROUGE_pegasus_Complete", do_save_to_json=True)

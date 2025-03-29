from datasets import load_dataset
# import sys

dataset_name = ""

if dataset_name == "wikihow":
    dataset = load_dataset(dataset_name, "all", "wikihowAll.csv")
    dataset = dataset.rename_column("headline","summary")
    dataset = dataset.rename_column("text","document")
else:
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")

    print(dataset.column_names)

    dataset = dataset.rename_column("highlights","summary")
    dataset = dataset.rename_column("article","document")

    print(dataset.column_names)

# dataset.save_to_disk("cnn_dailymail_raw")

dataset["train"].save_to_disk("cnn_dailymail_raw/train")
dataset["validation"].save_to_disk("cnn_dailymail_raw/validation")
dataset["test"].save_to_disk("cnn_dailymail_raw/test")


"""dataset = load_dataset("EdinburghNLP/xsum")

print(dataset.column_names)

# Search for a column or a row named 'length' in the dataset
if "length" in dataset.column_names:
    print(dataset["length"])

dataset.save_to_disk("xsum_raw")"""

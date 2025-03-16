from datasets import load_dataset
import sys

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

dataset.save_to_disk("cnn_dailymail_raw")

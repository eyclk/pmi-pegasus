from datasets import Dataset
# import pandas as pd


ds = Dataset.from_file("../xsum_comb/test/dataset.arrow")  # cnn_comb/test/data-00000-of-00001.arrow

pd_ds = ds.to_pandas()

"""
print(pd_ds["document"][3], "\n\n")  ### View a single example from the TEST dataset
print(pd_ds["summary"][3], "\n\n")
"""

print(pd_ds.head(), "\n", pd_ds.columns, "\n", pd_ds.shape, "\n\n")

# Search for a data storage area inside the dataset named "document_ents"
# print(pd_ds["document"][1], "\n\n")  # ['document_ents']

ds.features.output_all_columns = True
print(ds.format, "\n", ds.features, "\n\n")  ## ====> Failed to find 'document_ents' in the dataset


# Print first 5 observed examples from the dataset
for i in range(5):
    print(pd_ds["document"][i], "\n", pd_ds["id"][i], "\n\n", pd_ds["summary"][i], "\n\n")

    print("====> document_ents: ", pd_ds["document_ents"][i], "\n\n")
    print("====> summary_ents: ", pd_ds["summary_ents"][i], "\n\n")
    print("-"*50, "\n\n")

""" # Print the details of the example with the given 'summary', e.g., On the first day in his new job, Choe Peng Sum was given a fairly simple brief: "Just go make us a lot of money."
selected_example = pd_ds[pd_ds["summary"] == "US tennis star Venus Williams has been involved in a car accident that led to the death of a 78-year-old man."]

# Get index of selected_example
selected_example_index = selected_example.index[0]

print(pd_ds["document"][selected_example_index], "\n", pd_ds["id"][selected_example_index], "\n\n", pd_ds["summary"][selected_example_index])"""

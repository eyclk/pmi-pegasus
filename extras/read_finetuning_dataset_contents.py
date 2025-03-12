from datasets import Dataset
import pandas as pd


ds = Dataset.from_file("test_xsum/dataset.arrow")

pd_ds = ds.to_pandas()

print(pd_ds.head(), "\n", pd_ds.columns, "\n", pd_ds.shape, "\n\n")

# Index of observed example from the dataset
index = 2
print(pd_ds["document"][index], "\n", pd_ds["id"][index], "\n\n", pd_ds["summary"][index])

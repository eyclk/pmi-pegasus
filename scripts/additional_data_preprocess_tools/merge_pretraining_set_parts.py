import datasets


dataset_parts_folder = "../../PREPROCESSED_DATASETS/"

new_complete_dataset_folder = "../../PREPROCESSED_DATASETS/COMPLETE_c4_realnewslike_processed_PMI/"


"""example_set_name = dataset_parts_folder + "c4_realnewslike_processed_PMI_9_to_10_mil_combined"

# Load the example set part. Only train set. It is an arrow file.
example_set = datasets.load_from_disk(example_set_name)["train"]

# Print the first 10 rows line by line.
print("First 10 rows of the example set:")
for i in range(10):
    print(example_set[i])"""

all_dataset_parts = ["c4_realnewslike_processed_PMI_1_mil_combined",
                     "c4_realnewslike_processed_PMI_1_to_2_mil_combined",
                     "c4_realnewslike_processed_PMI_2_to_3_mil_combined",
                     "c4_realnewslike_processed_PMI_3_to_4_mil_combined",
                     "c4_realnewslike_processed_PMI_4_to_5_mil_combined",
                     "c4_realnewslike_processed_PMI_5_to_6_mil_combined",
                     "c4_realnewslike_processed_PMI_6_to_7_mil_combined",
                     "c4_realnewslike_processed_PMI_7_to_8_mil_combined",
                     "c4_realnewslike_processed_PMI_8_to_9_mil_combined",
                     "c4_realnewslike_processed_PMI_9_to_10_mil_combined",
                     "c4_realnewslike_processed_PMI_10_to_11_mil_combined",
                     "c4_realnewslike_processed_PMI_11_to_12_mil_combined",
                     "c4_realnewslike_processed_PMI_12_to_13_mil_combined",
                     "c4_realnewslike_processed_PMI_13_to_14_mil_combined"]

# Add main folder name in front of each dataset part
all_dataset_parts = [dataset_parts_folder + part for part in all_dataset_parts]

# Load all dataset parts in for loop and concatenate them into a single dataset
complete_dataset = []
for dataset_part in all_dataset_parts:
    print(f"\nLoading dataset part: {dataset_part}")
    dataset = datasets.load_from_disk(dataset_part)["train"]
    complete_dataset.append(dataset)

# Concatenate all dataset parts into a single dataset
complete_dataset_object = datasets.concatenate_datasets(complete_dataset)

# Save the complete dataset to disk as an arrow file. Similar to each part, it should be saved as train split of the dataset.
print("\nSaving the complete dataset to disk...")
complete_dataset_object.save_to_disk(new_complete_dataset_folder + "train")

# Open a new file called dataset_dict.json and write the following content to it: "{"splits": ["train"]}"
with open(new_complete_dataset_folder + "dataset_dict.json", "w") as f:
    f.write('{"splits": ["train"]}')


# Print the number of rows in the complete dataset
print(f"\nNumber of rows in the complete dataset: {len(complete_dataset_object)}")

# Print the first 10 rows of the complete dataset
print("\nFirst 10 rows of the complete dataset:")
for i in range(10):
    print(complete_dataset_object[i])

# Print the column names of the complete dataset
print("\nColumn names of the complete dataset:")
print(complete_dataset_object.column_names)

import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModel
AutoModel.from_pretrained("roberta-large")  # , force_download=True   # Download the model early to avoid errors when importing bert_score

#  from qafacteval import QAFactEval    # -------------> FAILED TO INSTALL LIBRARY DUE TO DEPENDENCY ISSUES (TORCH MUST BE EXACTLY 1.6.0, BUT CONFLICTS WITH BERTSCORE)
# import torch


test_dataset_path = "xsum_comb/test/dataset.arrow"  # Path to the test dataset
generated_predictions_file_path = "temp_xsum_generated_predictions_from_rouge_model.txt"  # Path to the generated predictions file
metrics_path = "all_result_metrics.csv"  # Path to save the evaluation metrics


# Function to compute ROUGE scores
def compute_rouge(target_summary, predicted_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(target_summary, predicted_summary)

    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }


# Load test dataset
#  df = pd.read_csv("test_dataset.csv")

ds = Dataset.from_file(test_dataset_path)
pd_ds = ds.to_pandas()

# Take "document" and "summary" columns from pd_ds. Rename them to "document" and "target_summary" and store them inside df
df = pd_ds[["document", "summary"]].rename(columns={"summary": "target_summary"})

# Read the predicted summaries from a txt file named "temp_xsum_generated_predictions_from_rouge_model.txt" and add them to df as a new column "predicted_summary"
with open(generated_predictions_file_path, "r") as f:
    predicted_summaries = f.readlines()
predicted_summaries = [line.strip() for line in predicted_summaries]
df["predicted_summary"] = predicted_summaries

# Check if the number of predicted summaries matches the number of rows in df
if len(predicted_summaries) != len(df):
    raise ValueError("The number of predicted summaries does not match the number of rows in the DataFrame.")

# Check if the predicted summaries are not empty
if any(len(summary) == 0 for summary in predicted_summaries):
    raise ValueError("One or more predicted summaries are empty.")

# Check if the target summaries are not empty
if any(len(summary) == 0 for summary in df["target_summary"]):
    raise ValueError("One or more target summaries are empty.")


print("\n--> Dataset and predicted summaries have been read...\n")


# Compute ROUGE scores
rouge_results = df.apply(lambda row: compute_rouge(row["target_summary"], row["predicted_summary"]), axis=1)
rouge_df = pd.DataFrame(rouge_results.tolist())

print("\n--> ROUGE scores have been computed...\n")


# Compute BERTScore
"""P, R, F1 = bert_score(df["predicted_summary"].tolist(), df["target_summary"].tolist(), lang="en", rescale_with_baseline=True)
bert_df = pd.DataFrame({"bert_precision": P.tolist(), "bert_recall": R.tolist(), "bert_f1": F1.tolist()})"""

batch_size = 16  # Adjust based on your GPU memory
bert_scores = {"bert_precision": [], "bert_recall": [], "bert_f1": []}

for i in tqdm(range(0, len(df), batch_size), desc="Calculating BERTScore"):
    batch_preds = df["predicted_summary"].tolist()[i: i + batch_size]
    batch_refs = df["target_summary"].tolist()[i: i + batch_size]

    P, R, F1 = bert_score(batch_preds, batch_refs, lang="en", rescale_with_baseline=True)

    bert_scores["bert_precision"].extend(P.tolist())
    bert_scores["bert_recall"].extend(R.tolist())
    bert_scores["bert_f1"].extend(F1.tolist())

bert_df = pd.DataFrame(bert_scores)

print("--> BERTScore has been computed...\n")


# Initialize QAFactEval
"""device = "cuda" if torch.cuda.is_available() else "cpu"
qafe = QAFactEval(device=device, use_lerc_quip=True, generation_batch_size=8)

# Compute QAFactEval scores
qafacteval_scores = qafe.compute_scores(
    df["document"].tolist(),
    df["predicted_summary"].tolist()
)
qafacteval_df = pd.DataFrame({"qafacteval_score": qafacteval_scores})"""


# Combine all results
df_results = pd.concat([df, rouge_df, bert_df], axis=1)  # , qafacteval_df

# Save results to CSV
df_results.to_csv(metrics_path, index=False)

# Compute and print average scores
avg_scores = df_results.iloc[:, 3:].mean()
print("\nAverage Evaluation Scores:")
print(avg_scores)

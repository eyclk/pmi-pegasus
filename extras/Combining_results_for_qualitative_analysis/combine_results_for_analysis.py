import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from datasets import Dataset
from transformers import AutoModel
from tqdm import tqdm
from transformers import logging

# Set the logging level to ERROR to suppress warnings
logging.set_verbosity_error()

AutoModel.from_pretrained("roberta-large")  # , force_download=True   # Download the model early to avoid errors when importing bert_score


# Function to compute ROUGE scores
def compute_rouge(target_summary, predicted_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(target_summary, predicted_summary)

    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
}


def combine_results_of_xsum():

    batch_size = 32

    test_dataset_path = "xsum_result_files/test_set_xsum/dataset.arrow"
    pmi_generated_predictions_file_path = "xsum_result_files/pmi_pegasus_xsum_generated_summaries/generated_predictions.txt"
    rouge_generated_predictions_file_path = "xsum_result_files/rouge_pegasus_xsum_generated_summaries/generated_predictions.txt"

    combined_output_path = "xsum_result_files/xsum_combined_results_for_analysis.json"


    # Load test dataset
    ds = Dataset.from_file(test_dataset_path)
    pd_ds = ds.to_pandas()

    # Acquire the target summaries from the test dataset
    target_summaries = pd_ds["summary"].tolist()

    # Read the predicted summaries from the PMI and ROUGE files
    with open(pmi_generated_predictions_file_path, "r", encoding="utf-8") as f:
        pmi_generated_summaries = f.readlines()
    pmi_generated_summaries = [line.strip() for line in pmi_generated_summaries]
    with open(rouge_generated_predictions_file_path, "r", encoding="utf-8") as f:
        rouge_generated_summaries = f.readlines()
    rouge_generated_summaries = [line.strip() for line in rouge_generated_summaries]

    # Check if the number of predicted summaries matches the number of rows in pd_ds
    if len(pmi_generated_summaries) != len(pd_ds):
        raise ValueError("The number of PMI generated summaries does not match the number of rows in the DataFrame.")
    if len(rouge_generated_summaries) != len(pd_ds):
        raise ValueError("The number of ROUGE generated summaries does not match the number of rows in the DataFrame.")


    all_target_summaries = []
    all_pmi_predicted_summaries = []
    all_rouge_predicted_summaries = []
    all_pmi_rouge1_scores = []
    all_pmi_bert_scores = []
    all_rouge_rouge1_scores = []
    all_rouge_bert_scores = []

    # Iterate through the test dataset and calculate the scores
    """for i in tqdm(range(len(pd_ds)), desc="Calculating scores"):
        target_summary = target_summaries[i]
        pmi_generated_summary = pmi_generated_summaries[i]
        rouge_generated_summary = rouge_generated_summaries[i]

        all_target_summaries.append(target_summary)
        all_pmi_predicted_summaries.append(pmi_generated_summary)
        all_rouge_predicted_summaries.append(rouge_generated_summary)

        # Calculate ROUGE1 F1 scores for the PMI generated summary
        rouge_scores = compute_rouge(target_summary, pmi_generated_summary)
        all_pmi_rouge1_scores.append(rouge_scores["rouge1_f1"])

        # Calculate BERT F1 scores for the PMI generated summary
        pmi_bert_score = bert_score([pmi_generated_summary], [target_summary], lang="en", model_type="roberta-large")
        all_pmi_bert_scores.append(pmi_bert_score[2].item())

        # Calculate ROUGE1 F1 scores for the ROUGE generated summary
        rouge_scores = compute_rouge(target_summary, rouge_generated_summary)
        all_rouge_rouge1_scores.append(rouge_scores["rouge1_f1"])

        # Calculate BERT F1 scores for the ROUGE generated summary
        rouge_bert_score = bert_score([rouge_generated_summary], [target_summary], lang="en", model_type="roberta-large")
        all_rouge_bert_scores.append(rouge_bert_score[2].item())"""

    for i in tqdm(range(0, len(pd_ds), batch_size), desc="Calculating scores"):
        batch_target_summaries = target_summaries[i:i + batch_size]
        batch_pmi_generated_summaries = pmi_generated_summaries[i:i + batch_size]
        batch_rouge_generated_summaries = rouge_generated_summaries[i:i + batch_size]

        all_target_summaries.extend(batch_target_summaries)
        all_pmi_predicted_summaries.extend(batch_pmi_generated_summaries)
        all_rouge_predicted_summaries.extend(batch_rouge_generated_summaries)

        # Calculate ROUGE1 F1 scores for the PMI generated summaries
        for target, pmi_summary in zip(batch_target_summaries, batch_pmi_generated_summaries):
            rouge_scores = compute_rouge(target, pmi_summary)
            all_pmi_rouge1_scores.append(rouge_scores["rouge1_f1"])

        # Calculate BERT F1 scores for the PMI generated summaries
        pmi_bert_scores = bert_score(batch_pmi_generated_summaries, batch_target_summaries, lang="en",
                                     model_type="roberta-large")
        all_pmi_bert_scores.extend(pmi_bert_scores[2].tolist())

        # Calculate ROUGE1 F1 scores for the ROUGE generated summaries
        for target, rouge_summary in zip(batch_target_summaries, batch_rouge_generated_summaries):
            rouge_scores = compute_rouge(target, rouge_summary)
            all_rouge_rouge1_scores.append(rouge_scores["rouge1_f1"])

        # Calculate BERT F1 scores for the ROUGE generated summaries
        rouge_bert_scores = bert_score(batch_rouge_generated_summaries, batch_target_summaries, lang="en",
                                       model_type="roberta-large")
        all_rouge_bert_scores.extend(rouge_bert_scores[2].tolist())


    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        "ground_truth_summary": all_target_summaries,
        "pmi_pegasus_generated_summary": all_pmi_predicted_summaries,
        "rouge_pegasus_generated_summary": all_rouge_predicted_summaries,
        "pmi_pegasus_rouge1_score": all_pmi_rouge1_scores,
        "rouge_pegasus_rouge1_score": all_rouge_rouge1_scores,
        "pmi_pegasus_bert_score": all_pmi_bert_scores,
        "rouge_pegasus_bert_score": all_rouge_bert_scores
    })

    # Save the DataFrame to a JSON file with pretty formatting
    results_df.to_json(combined_output_path, orient="records", lines=True, indent=4)
    print(f"Combined results saved to {combined_output_path}")


def combine_results_of_cnn():

    batch_size = 32

    test_dataset_path = "cnn_result_files/test_set_cnn/data-00000-of-00001.arrow"
    pmi_generated_predictions_file_path = "cnn_result_files/pmi_pegasus_cnn_generated_summaries/generated_predictions.txt"
    rouge_generated_predictions_file_path = "cnn_result_files/rouge_pegasus_cnn_generated_summaries/generated_predictions.txt"

    combined_output_path = "cnn_result_files/cnn_combined_results_for_analysis.json"


    # Load test dataset
    ds = Dataset.from_file(test_dataset_path)
    pd_ds = ds.to_pandas()

    # Acquire the target summaries from the test dataset
    target_summaries = pd_ds["summary"].tolist()

    # Read the predicted summaries from the PMI and ROUGE files
    with open(pmi_generated_predictions_file_path, "r", encoding="utf-8") as f:
        pmi_generated_summaries = f.readlines()
    pmi_generated_summaries = [line.strip() for line in pmi_generated_summaries]
    with open(rouge_generated_predictions_file_path, "r", encoding="utf-8") as f:
        rouge_generated_summaries = f.readlines()
    rouge_generated_summaries = [line.strip() for line in rouge_generated_summaries]

    # Check if the number of predicted summaries matches the number of rows in pd_ds
    if len(pmi_generated_summaries) != len(pd_ds):
        raise ValueError("The number of PMI generated summaries does not match the number of rows in the DataFrame.")
    if len(rouge_generated_summaries) != len(pd_ds):
        raise ValueError("The number of ROUGE generated summaries does not match the number of rows in the DataFrame.")


    all_target_summaries = []
    all_pmi_predicted_summaries = []
    all_rouge_predicted_summaries = []
    all_pmi_rouge1_scores = []
    all_pmi_bert_scores = []
    all_rouge_rouge1_scores = []
    all_rouge_bert_scores = []

    # Iterate through the test dataset and calculate the scores
    """for i in tqdm(range(len(pd_ds)), desc="Calculating scores"):
        target_summary = target_summaries[i]
        pmi_generated_summary = pmi_generated_summaries[i]
        rouge_generated_summary = rouge_generated_summaries[i]

        all_target_summaries.append(target_summary)
        all_pmi_predicted_summaries.append(pmi_generated_summary)
        all_rouge_predicted_summaries.append(rouge_generated_summary)

        # Calculate ROUGE1 F1 scores for the PMI generated summary
        rouge_scores = compute_rouge(target_summary, pmi_generated_summary)
        all_pmi_rouge1_scores.append(rouge_scores["rouge1_f1"])

        # Calculate BERT F1 scores for the PMI generated summary
        pmi_bert_score = bert_score([pmi_generated_summary], [target_summary], lang="en", model_type="roberta-large")
        all_pmi_bert_scores.append(pmi_bert_score[2].item())

        # Calculate ROUGE1 F1 scores for the ROUGE generated summary
        rouge_scores = compute_rouge(target_summary, rouge_generated_summary)
        all_rouge_rouge1_scores.append(rouge_scores["rouge1_f1"])

        # Calculate BERT F1 scores for the ROUGE generated summary
        rouge_bert_score = bert_score([rouge_generated_summary], [target_summary], lang="en",
                                      model_type="roberta-large")
        all_rouge_bert_scores.append(rouge_bert_score[2].item())"""

    for i in tqdm(range(0, len(pd_ds), batch_size), desc="Calculating scores"):
        batch_target_summaries = target_summaries[i:i + batch_size]
        batch_pmi_generated_summaries = pmi_generated_summaries[i:i + batch_size]
        batch_rouge_generated_summaries = rouge_generated_summaries[i:i + batch_size]

        all_target_summaries.extend(batch_target_summaries)
        all_pmi_predicted_summaries.extend(batch_pmi_generated_summaries)
        all_rouge_predicted_summaries.extend(batch_rouge_generated_summaries)

        # Calculate ROUGE1 F1 scores for the PMI generated summaries
        for target, pmi_summary in zip(batch_target_summaries, batch_pmi_generated_summaries):
            rouge_scores = compute_rouge(target, pmi_summary)
            all_pmi_rouge1_scores.append(rouge_scores["rouge1_f1"])

        # Calculate BERT F1 scores for the PMI generated summaries
        pmi_bert_scores = bert_score(batch_pmi_generated_summaries, batch_target_summaries, lang="en",
                                     model_type="roberta-large")
        all_pmi_bert_scores.extend(pmi_bert_scores[2].tolist())

        # Calculate ROUGE1 F1 scores for the ROUGE generated summaries
        for target, rouge_summary in zip(batch_target_summaries, batch_rouge_generated_summaries):
            rouge_scores = compute_rouge(target, rouge_summary)
            all_rouge_rouge1_scores.append(rouge_scores["rouge1_f1"])

        # Calculate BERT F1 scores for the ROUGE generated summaries
        rouge_bert_scores = bert_score(batch_rouge_generated_summaries, batch_target_summaries, lang="en",
                                       model_type="roberta-large")
        all_rouge_bert_scores.extend(rouge_bert_scores[2].tolist())


    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        "ground_truth_summary": all_target_summaries,
        "pmi_pegasus_generated_summary": all_pmi_predicted_summaries,
        "rouge_pegasus_generated_summary": all_rouge_predicted_summaries,
        "pmi_pegasus_rouge1_score": all_pmi_rouge1_scores,
        "rouge_pegasus_rouge1_score": all_rouge_rouge1_scores,
        "pmi_pegasus_bert_score": all_pmi_bert_scores,
        "rouge_pegasus_bert_score": all_rouge_bert_scores
    })

    # Save the DataFrame to a JSON file with pretty formatting
    results_df.to_json(combined_output_path, orient="records", lines=True, indent=4)
    print(f"Combined results saved to {combined_output_path}")


if __name__ == "__main__":

    # combine_results_of_xsum()

    combine_results_of_cnn()

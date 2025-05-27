# import pandas as pd
# from rouge_score import rouge_scorer
# from bert_score import score as bert_score
from datasets import Dataset
# from transformers import AutoModel
from tqdm import tqdm
# from transformers import logging
import json

from qafacteval import QAFactEval    # -------------> FAILED TO INSTALL LIBRARY DUE TO DEPENDENCY ISSUES (TORCH MUST BE EXACTLY 1.6.0, BUT CONFLICTS WITH BERTSCORE)
# import torch                        ## ------------> Created a new virtual environment named "qaeval" only for this task, with torch==1.6.0 and qafacteval installed.
######## --------------------> This also failed due to a package named edlib not being pre-built for windows. So, I had to do all of this on Linux.

# Set the logging level to ERROR to suppress warnings
# logging.set_verbosity_error()

qafe = QAFactEval(gpu=True)


def calc_qaeval_metric_of_xsum():

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
    """all_pmi_rouge1_scores = []
    all_pmi_bert_scores = []
    all_rouge_rouge1_scores = []
    all_rouge_bert_scores = []"""

    all_pmi_qaeval_scores = []
    all_rouge_qaeval_scores = []

    for i in tqdm(range(0, len(pd_ds), batch_size), desc="Calculating QAeval scores"):
        batch_target_summaries = target_summaries[i:i + batch_size]
        batch_pmi_generated_summaries = pmi_generated_summaries[i:i + batch_size]
        batch_rouge_generated_summaries = rouge_generated_summaries[i:i + batch_size]

        all_target_summaries.extend(batch_target_summaries)
        all_pmi_predicted_summaries.extend(batch_pmi_generated_summaries)
        all_rouge_predicted_summaries.extend(batch_rouge_generated_summaries)

        # Compute QAFactEval scores
        qafacteval_scores_pmi_model = qafe.compute_scores(
            batch_target_summaries,
            batch_pmi_generated_summaries  # .tolist()
        )
        all_pmi_qaeval_scores.extend(qafacteval_scores_pmi_model["scores"])

        qafacteval_scores_rouge_model = qafe.compute_scores(
            batch_target_summaries,
            batch_rouge_generated_summaries  # .tolist()
        )
        all_rouge_qaeval_scores.extend(qafacteval_scores_rouge_model["scores"])

        # Calculate ROUGE1 F1 scores for the PMI generated summaries
        """for target, pmi_summary in zip(batch_target_summaries, batch_pmi_generated_summaries):
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
        all_rouge_bert_scores.extend(rouge_bert_scores[2].tolist())"""


    # Open the combined output file in write mode. Then, add each model's QAeval scores to the file. For each dictionary in the list stored in the combined results file, add the QAeval scores for both models inside the dictionary.
    with open(combined_output_path, "r", encoding="utf-8") as f:
        combined_results = json.load(f)
    for i, result in enumerate(combined_results):
        # Check if the result dictionary has the expected keys, specifically compare ground_truth_summary
        if result["ground_truth_summary"] != all_target_summaries[i]:
            raise ValueError(f"Mismatch in ground truth summary at index {i}. Expected: {all_target_summaries[i]}, Found: {result['ground_truth_summary']}")

        result["pmi_pegasus_qaeval_score"] = all_pmi_qaeval_scores[i]
        result["rouge_pegasus_qaeval_score"] = all_rouge_qaeval_scores[i]

    # Save the updated combined results back to the file
    with open(combined_output_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4)

    # Print a message indicating that the QAeval scores have been added to the already existing combined output file
    print(f"QAeval scores added to already existing {combined_output_path}")


def calc_qaeval_metric_of_cnn():

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
        raise ValueError(
            "The number of PMI generated summaries does not match the number of rows in the DataFrame.")
    if len(rouge_generated_summaries) != len(pd_ds):
        raise ValueError(
            "The number of ROUGE generated summaries does not match the number of rows in the DataFrame.")

    all_target_summaries = []
    all_pmi_predicted_summaries = []
    all_rouge_predicted_summaries = []
    """all_pmi_rouge1_scores = []
    all_pmi_bert_scores = []
    all_rouge_rouge1_scores = []
    all_rouge_bert_scores = []"""

    all_pmi_qaeval_scores = []
    all_rouge_qaeval_scores = []

    for i in tqdm(range(0, len(pd_ds), batch_size), desc="Calculating QAeval scores"):
        batch_target_summaries = target_summaries[i:i + batch_size]
        batch_pmi_generated_summaries = pmi_generated_summaries[i:i + batch_size]
        batch_rouge_generated_summaries = rouge_generated_summaries[i:i + batch_size]

        all_target_summaries.extend(batch_target_summaries)
        all_pmi_predicted_summaries.extend(batch_pmi_generated_summaries)
        all_rouge_predicted_summaries.extend(batch_rouge_generated_summaries)

        # Compute QAFactEval scores
        qafacteval_scores_pmi_model = qafe.compute_scores(
            batch_target_summaries,
            batch_pmi_generated_summaries  # .tolist()
        )
        all_pmi_qaeval_scores.extend(qafacteval_scores_pmi_model["scores"])

        qafacteval_scores_rouge_model = qafe.compute_scores(
            batch_target_summaries,
            batch_rouge_generated_summaries  # .tolist()
        )
        all_rouge_qaeval_scores.extend(qafacteval_scores_rouge_model["scores"])

        # Calculate ROUGE1 F1 scores for the PMI generated summaries
        """for target, pmi_summary in zip(batch_target_summaries, batch_pmi_generated_summaries):
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
        all_rouge_bert_scores.extend(rouge_bert_scores[2].tolist())"""

    # Open the combined output file in write mode. Then, add each model's QAeval scores to the file. For each dictionary in the list stored in the combined results file, add the QAeval scores for both models inside the dictionary.
    with open(combined_output_path, "r", encoding="utf-8") as f:
        combined_results = json.load(f)
    for i, result in enumerate(combined_results):
        # Check if the result dictionary has the expected keys, specifically compare ground_truth_summary
        if result["ground_truth_summary"] != all_target_summaries[i]:
            raise ValueError(
                f"Mismatch in ground truth summary at index {i}. Expected: {all_target_summaries[i]}, Found: {result['ground_truth_summary']}")

        result["pmi_pegasus_qaeval_score"] = all_pmi_qaeval_scores[i]
        result["rouge_pegasus_qaeval_score"] = all_rouge_qaeval_scores[i]

    # Save the updated combined results back to the file
    with open(combined_output_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4)

    # Print a message indicating that the QAeval scores have been added to the already existing combined output file
    print(f"QAeval scores added to already existing {combined_output_path}")


if __name__ == "__main__":

    calc_qaeval_metric_of_xsum()

    #  calc_qaeval_metric_of_cnn()


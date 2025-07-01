# import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import json
from sacrerouge.metrics import QAEval

# from qafacteval import QAFactEval    # -------------> FAILED TO INSTALL LIBRARY DUE TO DEPENDENCY ISSUES (TORCH MUST BE EXACTLY 1.6.0, BUT CONFLICTS WITH BERTSCORE)
# import torch                        ## ------------> Created a new virtual environment named "qaeval" only for this task, with torch==1.6.0 and qafacteval installed.
######## --------------------> This also failed due to a package named edlib not being pre-built for windows. So, I had to do all of this on Linux.
########### ABOVE NOTES ARE NOT VALID ANYMORE. I FIXED MOST OF THE ISSUES AND NOW THE LIBRARY WORKS FINE.


qa_metric = QAEval(cuda_device=0)  # Set to -1 if you want to run on CPU


def calc_qaeval_metric_of_xsum():
    batch_size = 32

    test_dataset_path = "xsum_result_files/test_set_xsum/dataset.arrow"
    pmi_generated_predictions_file_path = "xsum_result_files/pmi_pegasus_xsum_generated_summaries/generated_predictions.txt"
    rouge_generated_predictions_file_path = "xsum_result_files/rouge_pegasus_xsum_generated_summaries/generated_predictions.txt"

    combined_output_path = "xsum_result_files/xsum_combined_results_for_analysis.json"
    combined_output_path_with_qaeval = "xsum_result_files/xsum_combined_results_for_analysis__with_qaeval.json"

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

    all_pmi_qaeval_f1_scores = []
    all_rouge_qaeval_f1_scores = []
    all_pmi_qaeval_scores_is_answered = []
    all_rouge_qaeval_scores_is_answered = []

    for i in tqdm(range(0, len(pd_ds), batch_size), desc="Calculating QAeval scores for XSUM dataset"):
        batch_target_summaries = target_summaries[i:i + batch_size]
        batch_pmi_generated_summaries = pmi_generated_summaries[i:i + batch_size]
        batch_rouge_generated_summaries = rouge_generated_summaries[i:i + batch_size]

        all_target_summaries.extend(batch_target_summaries)
        all_pmi_predicted_summaries.extend(batch_pmi_generated_summaries)
        all_rouge_predicted_summaries.extend(batch_rouge_generated_summaries)

        # Each item in batch_target_summaries should be a list containing a single string, so we convert it to a list of lists
        batch_target_summaries_for_metric = [[summary] for summary in batch_target_summaries]

        # Compute QAEval scores
        qaeval_scores_pmi_model = qa_metric.score_all(
            batch_pmi_generated_summaries,
            batch_target_summaries_for_metric
        )

        qaeval_scores_pmi_model_f1 = [s["qa-eval"]["f1"] for s in
                                           qaeval_scores_pmi_model]  # We choose to only keep the F1 score from the QAEval scores
        all_pmi_qaeval_f1_scores.extend(qaeval_scores_pmi_model_f1)

        qaeval_scores_pmi_model_is_answered = [s["qa-eval"]["is_answered"] for s in
                                                  qaeval_scores_pmi_model]
        all_pmi_qaeval_scores_is_answered.extend(qaeval_scores_pmi_model_is_answered)

        qaeval_scores_rouge_model = qa_metric.score_all(
            batch_rouge_generated_summaries,
            batch_target_summaries_for_metric
        )

        qaeval_scores_rouge_model_f1 = [s["qa-eval"]["f1"] for s in
                                             qaeval_scores_rouge_model]  # We choose to only keep the F1 score from the QAEval scores
        all_rouge_qaeval_f1_scores.extend(qaeval_scores_rouge_model_f1)

        qaeval_scores_rouge_model_is_answered = [s["qa-eval"]["is_answered"] for s in
                                               qaeval_scores_rouge_model]
        all_rouge_qaeval_scores_is_answered.extend(qaeval_scores_rouge_model_is_answered)

    # Open the combined output file in write mode. Then, add each model's QAeval scores to the file. For each dictionary in the list stored in the combined results file, add the QAeval scores for both models inside the dictionary.
    with open(combined_output_path, "r", encoding="utf-8") as f:
        combined_results = json.load(f)
    for i, result in enumerate(combined_results):
        # Check if the result dictionary has the expected keys, specifically compare ground_truth_summary
        if result["ground_truth_summary"] != all_target_summaries[i]:
            raise ValueError(
                f"Mismatch in ground truth summary at index {i}. Expected: {all_target_summaries[i]}, Found: {result['ground_truth_summary']}")

        result["pmi_pegasus_qaeval_f1_score"] = all_pmi_qaeval_f1_scores[i]
        result["rouge_pegasus_qaeval_f1_score"] = all_rouge_qaeval_f1_scores[i]
        result["pmi_pegasus_qaeval_is_answered_score"] = all_pmi_qaeval_scores_is_answered[i]
        result["rouge_pegasus_qaeval_is_answered_score"] = all_rouge_qaeval_scores_is_answered[i]

    # Save the updated combined results to a new file
    with open(combined_output_path_with_qaeval, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4)

    # Print a message indicating that the new combined results file that contains QAeval scores have been written (xsum dataset)
    print(f"\nQAeval scores written to {combined_output_path_with_qaeval}")


def calc_qaeval_metric_of_cnn():
    batch_size = 32

    test_dataset_path = "cnn_result_files/test_set_cnn/data-00000-of-00001.arrow"
    pmi_generated_predictions_file_path = "cnn_result_files/pmi_pegasus_cnn_generated_summaries/generated_predictions.txt"
    rouge_generated_predictions_file_path = "cnn_result_files/rouge_pegasus_cnn_generated_summaries/generated_predictions.txt"

    combined_output_path = "cnn_result_files/cnn_combined_results_for_analysis.json"
    combined_output_path_with_qaeval = "cnn_result_files/cnn_combined_results_for_analysis__with_qaeval.json"

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

    all_pmi_qaeval_f1_scores = []
    all_rouge_qaeval_f1_scores = []
    all_pmi_qaeval_scores_is_answered = []
    all_rouge_qaeval_scores_is_answered = []

    for i in tqdm(range(0, len(pd_ds), batch_size), desc="Calculating QAeval scores for CNN dataset"):
        batch_target_summaries = target_summaries[i:i + batch_size]
        batch_pmi_generated_summaries = pmi_generated_summaries[i:i + batch_size]
        batch_rouge_generated_summaries = rouge_generated_summaries[i:i + batch_size]

        all_target_summaries.extend(batch_target_summaries)
        all_pmi_predicted_summaries.extend(batch_pmi_generated_summaries)
        all_rouge_predicted_summaries.extend(batch_rouge_generated_summaries)

        # Each item in batch_target_summaries should be a list containing a single string, so we convert it to a list of lists
        batch_target_summaries_for_metric = [[summary] for summary in batch_target_summaries]

        # Compute QAEval scores
        qaeval_scores_pmi_model = qa_metric.score_all(
            batch_pmi_generated_summaries,
            batch_target_summaries_for_metric
        )

        qaeval_scores_pmi_model_f1 = [s["qa-eval"]["f1"] for s in
                                      qaeval_scores_pmi_model]  # We choose to only keep the F1 score from the QAEval scores
        all_pmi_qaeval_f1_scores.extend(qaeval_scores_pmi_model_f1)

        qaeval_scores_pmi_model_is_answered = [s["qa-eval"]["is_answered"] for s in
                                               qaeval_scores_pmi_model]
        all_pmi_qaeval_scores_is_answered.extend(qaeval_scores_pmi_model_is_answered)

        qaeval_scores_rouge_model = qa_metric.score_all(
            batch_rouge_generated_summaries,
            batch_target_summaries_for_metric
        )

        qaeval_scores_rouge_model_f1 = [s["qa-eval"]["f1"] for s in
                                        qaeval_scores_rouge_model]  # We choose to only keep the F1 score from the QAEval scores
        all_rouge_qaeval_f1_scores.extend(qaeval_scores_rouge_model_f1)

        qaeval_scores_rouge_model_is_answered = [s["qa-eval"]["is_answered"] for s in
                                                 qaeval_scores_rouge_model]
        all_rouge_qaeval_scores_is_answered.extend(qaeval_scores_rouge_model_is_answered)

    # Open the combined output file in write mode. Then, add each model's QAeval scores to the file. For each dictionary in the list stored in the combined results file, add the QAeval scores for both models inside the dictionary.
    with open(combined_output_path, "r", encoding="utf-8") as f:
        combined_results = json.load(f)
    for i, result in enumerate(combined_results):
        # Check if the result dictionary has the expected keys, specifically compare ground_truth_summary
        if result["ground_truth_summary"] != all_target_summaries[i]:
            raise ValueError(
                f"Mismatch in ground truth summary at index {i}. Expected: {all_target_summaries[i]}, Found: {result['ground_truth_summary']}")

        result["pmi_pegasus_qaeval_f1_score"] = all_pmi_qaeval_f1_scores[i]
        result["rouge_pegasus_qaeval_f1_score"] = all_rouge_qaeval_f1_scores[i]
        result["pmi_pegasus_qaeval_is_answered_score"] = all_pmi_qaeval_scores_is_answered[i]
        result["rouge_pegasus_qaeval_is_answered_score"] = all_rouge_qaeval_scores_is_answered[i]

    # Save the updated combined results to a new file
    with open(combined_output_path_with_qaeval, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4)

        # Print a message indicating that the new combined results file that contains QAeval scores have been written (CNN dataset)
        print(f"\nQAeval scores written to {combined_output_path_with_qaeval}")


if __name__ == "__main__":

    calc_qaeval_metric_of_xsum()

    print("\n\n**********************************************************\n\n")

    calc_qaeval_metric_of_cnn()


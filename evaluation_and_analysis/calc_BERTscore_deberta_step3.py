import torch
from datasets import Dataset
#  from tqdm import tqdm
import json
import bert_score

#######  We chose to use DeBERTa for calculating the F1 metric, as it is shown to be the best BERTscore compatible model. #######


model_name = "microsoft/deberta-xlarge-mnli"  # Use a BERTScore-compatible model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32  # Adjust as needed for your GPU

eval_for_FactPEGASUS = False


def calc_deberta_f1_metric_of_xsum():

    test_dataset_path = "xsum_result_files/test_set_xsum/dataset.arrow"
    pmi_generated_predictions_file_path = "xsum_result_files/pmi_pegasus_xsum_generated_summaries/generated_predictions.txt"
    rouge_generated_predictions_file_path = "xsum_result_files/rouge_pegasus_xsum_generated_summaries/generated_predictions.txt"
    if eval_for_FactPEGASUS:
        factPegasus_generated_predictions_file_path = "xsum_result_files/factpegasus_public_xsum_generated_summaries/generated_predictions.txt"

    combined_output_path = "xsum_result_files/xsum_combined_results_for_analysis__step2.json"
    combined_output_path_with_deberta_score = "xsum_result_files/xsum_combined_results_for_analysis__step3.json"

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
    if eval_for_FactPEGASUS:
        with open(factPegasus_generated_predictions_file_path, "r", encoding="utf-8") as f:
            factPegasus_generated_summaries = f.readlines()
        factPegasus_generated_summaries = [line.strip() for line in factPegasus_generated_summaries]

    # Check if the number of predicted summaries matches the number of rows in pd_ds
    if len(pmi_generated_summaries) != len(pd_ds):
        raise ValueError("The number of PMI generated summaries does not match the number of rows in the DataFrame.")
    if len(rouge_generated_summaries) != len(pd_ds):
        raise ValueError("The number of ROUGE generated summaries does not match the number of rows in the DataFrame.")
    if eval_for_FactPEGASUS:
        if len(factPegasus_generated_summaries) != len(pd_ds):
            raise ValueError("The number of FactPEGASUS generated summaries does not match the number of rows in the DataFrame.")

    #  all_target_summaries = []
    """all_pmi_predicted_summaries = []
    all_rouge_predicted_summaries = []

    all_pmi_llm_f1_scores = []
    all_rouge_llm_f1_scores = []"""

    print("\nStarting to calculate BERTScore F1 metric with DeBERTa...")

    _, _, f1_pmi = bert_score.score(
        cands=pmi_generated_summaries,
        refs=target_summaries,
        model_type=model_name,
        device=device,
        batch_size=batch_size,
        lang="en",
        rescale_with_baseline=True  ## NEW
    )
    all_pmi_deberta_f1_scores = f1_pmi.tolist()

    _, _, f1_rouge = bert_score.score(
        cands=rouge_generated_summaries,
        refs=target_summaries,
        model_type=model_name,
        device=device,
        batch_size=batch_size,
        lang="en",
        rescale_with_baseline=True  ## NEW
    )
    all_rouge_deberta_f1_scores = f1_rouge.tolist()

    if eval_for_FactPEGASUS:
        _, _, f1_factPegasus = bert_score.score(
            cands=factPegasus_generated_summaries,
            refs=target_summaries,
            model_type=model_name,
            device=device,
            batch_size=batch_size,
            lang="en",
            rescale_with_baseline=True  ## NEW
        )
        all_factPegasus_deberta_f1_scores = f1_factPegasus.tolist()

    all_target_summaries = target_summaries

    """for i in tqdm(range(0, len(pd_ds), batch_size), desc="Calculating LLM scores for XSUM dataset"):
        batch_target_summaries = target_summaries[i:i + batch_size]
        batch_pmi_generated_summaries = pmi_generated_summaries[i:i + batch_size]
        batch_rouge_generated_summaries = rouge_generated_summaries[i:i + batch_size]

        all_target_summaries.extend(batch_target_summaries)
        all_pmi_predicted_summaries.extend(batch_pmi_generated_summaries)
        all_rouge_predicted_summaries.extend(batch_rouge_generated_summaries)

        # Compute BERTScore F1 for PMI model
        llm_scores_pmi_model = compute_llm_score_batch(batch_pmi_generated_summaries, batch_target_summaries)
        all_pmi_llm_f1_scores.extend(llm_scores_pmi_model)

        # Compute BERTScore F1 for ROUGE model
        llm_scores_rouge_model = compute_llm_score_batch(batch_rouge_generated_summaries, batch_target_summaries)
        all_rouge_llm_f1_scores.extend(llm_scores_rouge_model)"""

    # Open the combined output file in write mode. Then, add each model's deberta F1 scores to the file. For each dictionary in the list stored in the combined results file, add the deberta scores for both models inside the dictionary.
    with open(combined_output_path, "r", encoding="utf-8") as f:
        combined_results = json.load(f)
    for i, result in enumerate(combined_results):
        # Check if the result dictionary has the expected keys, specifically compare ground_truth_summary
        if result["ground_truth_summary"] != all_target_summaries[i]:
            raise ValueError(
                f"Mismatch in ground truth summary at index {i}. Expected: {all_target_summaries[i]}, Found: {result['ground_truth_summary']}")

        result["pmi_pegasus_deberta_f1_score"] = all_pmi_deberta_f1_scores[i]
        result["rouge_pegasus_deberta_f1_score"] = all_rouge_deberta_f1_scores[i]
        if eval_for_FactPEGASUS:
            result["factpegasus_public_deberta_f1_score"] = all_factPegasus_deberta_f1_scores[i]

    # Save the updated combined results to a new file
    with open(combined_output_path_with_deberta_score, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4)

    # Print a message indicating that the new combined results file that contains deberta F1 scores have been written (xsum dataset)
    print(f"\nThe BERTscore (with Deberta) F1 scores written to {combined_output_path_with_deberta_score}")

    # Print average F1 scores for both models
    avg_pmi_f1 = sum(all_pmi_deberta_f1_scores) / len(all_pmi_deberta_f1_scores)
    avg_rouge_f1 = sum(all_rouge_deberta_f1_scores) / len(all_rouge_deberta_f1_scores)

    print(f"\nAverage PMI DeBERTa F1 score for XSUM: {avg_pmi_f1}")
    print(f"Average ROUGE DeBERTa F1 score for XSUM: {avg_rouge_f1}")

    if eval_for_FactPEGASUS:
        avg_factPegasus_f1 = sum(all_factPegasus_deberta_f1_scores) / len(all_factPegasus_deberta_f1_scores)
        print(f"Average FactPEGASUS DeBERTa F1 score for XSUM: {avg_factPegasus_f1}")


def calc_deberta_f1_metric_of_cnn():

    test_dataset_path = "cnn_result_files/test_set_cnn/data-00000-of-00001.arrow"
    pmi_generated_predictions_file_path = "cnn_result_files/pmi_pegasus_cnn_generated_summaries/generated_predictions.txt"
    rouge_generated_predictions_file_path = "cnn_result_files/rouge_pegasus_cnn_generated_summaries/generated_predictions.txt"
    if eval_for_FactPEGASUS:
        factPegasus_generated_predictions_file_path = "cnn_result_files/factpegasus_public_cnn_generated_summaries/generated_predictions.txt"

    combined_output_path = "cnn_result_files/cnn_combined_results_for_analysis__step2.json"
    combined_output_path_with_deberta_score = "cnn_result_files/cnn_combined_results_for_analysis__step3.json"

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
    if eval_for_FactPEGASUS:
        with open(factPegasus_generated_predictions_file_path, "r", encoding="utf-8") as f:
            factPegasus_generated_summaries = f.readlines()
        factPegasus_generated_summaries = [line.strip() for line in factPegasus_generated_summaries]

    # Check if the number of predicted summaries matches the number of rows in pd_ds
    if len(pmi_generated_summaries) != len(pd_ds):
        raise ValueError("The number of PMI generated summaries does not match the number of rows in the DataFrame.")
    if len(rouge_generated_summaries) != len(pd_ds):
        raise ValueError("The number of ROUGE generated summaries does not match the number of rows in the DataFrame.")
    if eval_for_FactPEGASUS:
        if len(factPegasus_generated_summaries) != len(pd_ds):
            raise ValueError("The number of FactPEGASUS generated summaries does not match the number of rows in the DataFrame.")

    print("\nStarting to calculate BERTScore F1 metric with DeBERTa...")

    _, _, f1_pmi = bert_score.score(
        cands=pmi_generated_summaries,
        refs=target_summaries,
        model_type=model_name,
        device=device,
        batch_size=batch_size,
        lang="en",
        rescale_with_baseline=True  ## NEW
    )
    all_pmi_deberta_f1_scores = f1_pmi.tolist()

    _, _, f1_rouge = bert_score.score(
        cands=rouge_generated_summaries,
        refs=target_summaries,
        model_type=model_name,
        device=device,
        batch_size=batch_size,
        lang="en",
        rescale_with_baseline=True  ## NEW
    )
    all_rouge_deberta_f1_scores = f1_rouge.tolist()

    if eval_for_FactPEGASUS:
        _, _, f1_factPegasus = bert_score.score(
            cands=factPegasus_generated_summaries,
            refs=target_summaries,
            model_type=model_name,
            device=device,
            batch_size=batch_size,
            lang="en",
            rescale_with_baseline=True  ## NEW
        )
        all_factPegasus_deberta_f1_scores = f1_factPegasus.tolist()

    all_target_summaries = target_summaries

    """all_target_summaries = []
    all_pmi_predicted_summaries = []
    all_rouge_predicted_summaries = []

    all_pmi_llm_f1_scores = []
    all_rouge_llm_f1_scores = []

    for i in tqdm(range(0, len(pd_ds), batch_size), desc="Calculating LLM scores for CNN dataset"):
        batch_target_summaries = target_summaries[i:i + batch_size]
        batch_pmi_generated_summaries = pmi_generated_summaries[i:i + batch_size]
        batch_rouge_generated_summaries = rouge_generated_summaries[i:i + batch_size]

        all_target_summaries.extend(batch_target_summaries)
        all_pmi_predicted_summaries.extend(batch_pmi_generated_summaries)
        all_rouge_predicted_summaries.extend(batch_rouge_generated_summaries)

        # Compute BERTScore F1 for PMI model
        llm_scores_pmi_model = compute_llm_score_batch(batch_pmi_generated_summaries, batch_target_summaries)
        all_pmi_llm_f1_scores.extend(llm_scores_pmi_model)

        # Compute BERTScore F1 for ROUGE model
        llm_scores_rouge_model = compute_llm_score_batch(batch_rouge_generated_summaries, batch_target_summaries)
        all_rouge_llm_f1_scores.extend(llm_scores_rouge_model)"""

    # Open the combined output file in write mode. Then, add each model's deberta F1 scores to the file. For each dictionary in the list stored in the combined results file, add the deberta F1 scores for both models inside the dictionary.
    with open(combined_output_path, "r", encoding="utf-8") as f:
        combined_results = json.load(f)
    for i, result in enumerate(combined_results):
        # Check if the result dictionary has the expected keys, specifically compare ground_truth_summary
        if result["ground_truth_summary"] != all_target_summaries[i]:
            raise ValueError(
                f"Mismatch in ground truth summary at index {i}. Expected: {all_target_summaries[i]}, Found: {result['ground_truth_summary']}")

        result["pmi_pegasus_deberta_f1_score"] = all_pmi_deberta_f1_scores[i]
        result["rouge_pegasus_deberta_f1_score"] = all_rouge_deberta_f1_scores[i]
        if eval_for_FactPEGASUS:
            result["factpegasus_public_deberta_f1_score"] = all_factPegasus_deberta_f1_scores[i]

    # Save the updated combined results to a new file
    with open(combined_output_path_with_deberta_score, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4)

    # Print a message indicating that the new combined results file that contains deberta F1 scores have been written (cnn dataset)
    print(f"\nThe BERTscore (with Deberta) F1 scores written to {combined_output_path_with_deberta_score}")

    # Print average F1 scores for both models
    avg_pmi_f1 = sum(all_pmi_deberta_f1_scores) / len(all_pmi_deberta_f1_scores)
    avg_rouge_f1 = sum(all_rouge_deberta_f1_scores) / len(all_rouge_deberta_f1_scores)

    print(f"\nAverage PMI DeBERTa F1 score for CNN: {avg_pmi_f1}")
    print(f"Average ROUGE DeBERTa F1 score for CNN: {avg_rouge_f1}")

    if eval_for_FactPEGASUS:
        avg_factPegasus_f1 = sum(all_factPegasus_deberta_f1_scores) / len(all_factPegasus_deberta_f1_scores)
        print(f"Average FactPEGASUS DeBERTa F1 score for CNN: {avg_factPegasus_f1}")


def calc_deberta_f1_metric_of_wikihow():

    test_dataset_path = "wikihow_result_files/test_set_wikihow/dataset.arrow"
    pmi_generated_predictions_file_path = "wikihow_result_files/pmi_pegasus_wikihow_generated_summaries/generated_predictions.txt"
    rouge_generated_predictions_file_path = "wikihow_result_files/rouge_pegasus_wikihow_generated_summaries/generated_predictions.txt"

    combined_output_path = "wikihow_result_files/wikihow_combined_results_for_analysis__step2.json"
    combined_output_path_with_deberta_score = "wikihow_result_files/wikihow_combined_results_for_analysis__step3.json"

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

    #  all_target_summaries = []
    """all_pmi_predicted_summaries = []
    all_rouge_predicted_summaries = []

    all_pmi_llm_f1_scores = []
    all_rouge_llm_f1_scores = []"""

    print("\nStarting to calculate BERTScore F1 metric with DeBERTa...")

    _, _, f1_pmi = bert_score.score(
        cands=pmi_generated_summaries,
        refs=target_summaries,
        model_type=model_name,
        device=device,
        batch_size=batch_size,
        lang="en",
        rescale_with_baseline=True  ## NEW
    )
    all_pmi_deberta_f1_scores = f1_pmi.tolist()

    _, _, f1_rouge = bert_score.score(
        cands=rouge_generated_summaries,
        refs=target_summaries,
        model_type=model_name,
        device=device,
        batch_size=batch_size,
        lang="en",
        rescale_with_baseline=True  ## NEW
    )
    all_rouge_deberta_f1_scores = f1_rouge.tolist()

    all_target_summaries = target_summaries

    """for i in tqdm(range(0, len(pd_ds), batch_size), desc="Calculating LLM scores for WIKIHOW dataset"):
        batch_target_summaries = target_summaries[i:i + batch_size]
        batch_pmi_generated_summaries = pmi_generated_summaries[i:i + batch_size]
        batch_rouge_generated_summaries = rouge_generated_summaries[i:i + batch_size]

        all_target_summaries.extend(batch_target_summaries)
        all_pmi_predicted_summaries.extend(batch_pmi_generated_summaries)
        all_rouge_predicted_summaries.extend(batch_rouge_generated_summaries)

        # Compute BERTScore F1 for PMI model
        llm_scores_pmi_model = compute_llm_score_batch(batch_pmi_generated_summaries, batch_target_summaries)
        all_pmi_llm_f1_scores.extend(llm_scores_pmi_model)

        # Compute BERTScore F1 for ROUGE model
        llm_scores_rouge_model = compute_llm_score_batch(batch_rouge_generated_summaries, batch_target_summaries)
        all_rouge_llm_f1_scores.extend(llm_scores_rouge_model)"""

    # Open the combined output file in write mode. Then, add each model's deberta F1 scores to the file. For each dictionary in the list stored in the combined results file, add the deberta scores for both models inside the dictionary.
    with open(combined_output_path, "r", encoding="utf-8") as f:
        combined_results = json.load(f)
    for i, result in enumerate(combined_results):
        # Check if the result dictionary has the expected keys, specifically compare ground_truth_summary
        if result["ground_truth_summary"] != all_target_summaries[i]:
            raise ValueError(
                f"Mismatch in ground truth summary at index {i}. Expected: {all_target_summaries[i]}, Found: {result['ground_truth_summary']}")

        result["pmi_pegasus_deberta_f1_score"] = all_pmi_deberta_f1_scores[i]
        result["rouge_pegasus_deberta_f1_score"] = all_rouge_deberta_f1_scores[i]

    # Save the updated combined results to a new file
    with open(combined_output_path_with_deberta_score, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4)

    # Print a message indicating that the new combined results file that contains deberta F1 scores have been written (WIKIHOW dataset)
    print(f"\nThe BERTscore (with Deberta) F1 scores written to {combined_output_path_with_deberta_score}")

    # Print average F1 scores for both models
    avg_pmi_f1 = sum(all_pmi_deberta_f1_scores) / len(all_pmi_deberta_f1_scores)
    avg_rouge_f1 = sum(all_rouge_deberta_f1_scores) / len(all_rouge_deberta_f1_scores)

    print(f"\nAverage PMI DeBERTa F1 score for WIKIHOW: {avg_pmi_f1}")
    print(f"Average ROUGE DeBERTa F1 score for WIKIHOW: {avg_rouge_f1}")


if __name__ == "__main__":
    # Calculate deberta F1 metric for XSUM dataset
    calc_deberta_f1_metric_of_xsum()

    print("\n\n**********************************************************\n\n")

    # Calculate deberta F1 metric for CNN dataset
    calc_deberta_f1_metric_of_cnn()

    print("\n\n**********************************************************\n\n")

    # Calculate deberta F1 metric for WikiHow dataset
    calc_deberta_f1_metric_of_wikihow()

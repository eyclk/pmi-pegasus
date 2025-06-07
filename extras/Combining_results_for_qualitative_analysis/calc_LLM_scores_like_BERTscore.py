import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
import json


model_name = "meta-llama/Llama-2-7b-hf"   # "bert-base-uncased"  # "mistralai/Mistral-7B-v0.3"   # "meta-llama/Llama-2-7b-hf"

token = "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"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModel.from_pretrained(model_name, token=token).to(device)
model.eval()

batch_size = 32


def compute_llm_score_batch(candidates, references):
    """
    candidates: list of str
    references: list of str
    Returns: list of f1 scores (float) for each example
    """
    assert len(candidates) == len(references), "Candidates and references must be the same length"

    f1_results = []

    """for i in range(0, len(candidates), batch_size):
        batch_cands = candidates[i:i+batch_size]
        batch_refs = references[i:i+batch_size]"""

    # Tokenize
    inputs_cand = tokenizer(candidates, return_tensors="pt", truncation=True, padding=True).to(device)
    inputs_ref = tokenizer(references, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        cand_embs = model(**inputs_cand).last_hidden_state  # (batch, cand_len, hidden)
        ref_embs = model(**inputs_ref).last_hidden_state    # (batch, ref_len, hidden)

        for j in range(len(candidates)):
            cand_emb = cand_embs[j]  # (cand_len, hidden)
            ref_emb = ref_embs[j]    # (ref_len, hidden)
            cand_emb = F.normalize(cand_emb, p=2, dim=-1)
            ref_emb = F.normalize(ref_emb, p=2, dim=-1)
            sim_matrix = torch.matmul(cand_emb, ref_emb.T)
            precision = sim_matrix.max(dim=1).values.mean()
            recall = sim_matrix.max(dim=0).values.mean()
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_results.append(f1.item())

    return f1_results


def calc_llm_f1_metric_of_xsum():

    test_dataset_path = "xsum_result_files/test_set_xsum/dataset.arrow"
    pmi_generated_predictions_file_path = "xsum_result_files/pmi_pegasus_xsum_generated_summaries/generated_predictions.txt"
    rouge_generated_predictions_file_path = "xsum_result_files/rouge_pegasus_xsum_generated_summaries/generated_predictions.txt"

    combined_output_path = "xsum_result_files/xsum_combined_results_for_analysis__with_qaeval.json"
    combined_output_path_with_llm_score = "xsum_result_files/xsum_combined_results_for_analysis__with_llm_score.json"

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

    all_pmi_llm_f1_scores = []
    all_rouge_llm_f1_scores = []

    for i in tqdm(range(0, len(pd_ds), batch_size), desc="Calculating LLM scores for XSUM dataset"):
        batch_target_summaries = target_summaries[i:i + batch_size]
        batch_pmi_generated_summaries = pmi_generated_summaries[i:i + batch_size]
        batch_rouge_generated_summaries = rouge_generated_summaries[i:i + batch_size]

        all_target_summaries.extend(batch_target_summaries)
        all_pmi_predicted_summaries.extend(batch_pmi_generated_summaries)
        all_rouge_predicted_summaries.extend(batch_rouge_generated_summaries)

        # Compute LLM scores for PMI model
        llm_scores_pmi_model = compute_llm_score_batch(batch_pmi_generated_summaries, batch_target_summaries)

        all_pmi_llm_f1_scores.extend(llm_scores_pmi_model)


        # Compute LLM scores for ROUGE model
        llm_scores_rouge_model = compute_llm_score_batch(batch_rouge_generated_summaries, batch_target_summaries)

        all_rouge_llm_f1_scores.extend(llm_scores_rouge_model)

    # Open the combined output file in write mode. Then, add each model's custom LLM scores to the file. For each dictionary in the list stored in the combined results file, add the LLM scores for both models inside the dictionary.
    with open(combined_output_path, "r", encoding="utf-8") as f:
        combined_results = json.load(f)
    for i, result in enumerate(combined_results):
        # Check if the result dictionary has the expected keys, specifically compare ground_truth_summary
        if result["ground_truth_summary"] != all_target_summaries[i]:
            raise ValueError(
                f"Mismatch in ground truth summary at index {i}. Expected: {all_target_summaries[i]}, Found: {result['ground_truth_summary']}")

        result["pmi_pegasus_llm_f1_score"] = all_pmi_llm_f1_scores[i]
        result["rouge_pegasus_llm_f1_score"] = all_rouge_llm_f1_scores[i]

    # Save the updated combined results to a new file
    with open(combined_output_path_with_llm_score, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4)

    # Print a message indicating that the new combined results file that contains new custom LLM scores have been written (xsum dataset)
    print(f"\nThe new custom LLM F1 scores written to {combined_output_path_with_llm_score}")


def calc_llm_f1_metric_of_cnn():

    test_dataset_path = "cnn_result_files/test_set_cnn/data-00000-of-00001.arrow"
    pmi_generated_predictions_file_path = "cnn_result_files/pmi_pegasus_cnn_generated_summaries/generated_predictions.txt"
    rouge_generated_predictions_file_path = "cnn_result_files/rouge_pegasus_cnn_generated_summaries/generated_predictions.txt"

    combined_output_path = "cnn_result_files/cnn_combined_results_for_analysis__with_qaeval.json"
    combined_output_path_with_llm_score = "cnn_result_files/cnn_combined_results_for_analysis__with_llm_score.json"

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

    all_pmi_llm_f1_scores = []
    all_rouge_llm_f1_scores = []

    for i in tqdm(range(0, len(pd_ds), batch_size), desc="Calculating LLM scores for CNN dataset"):
        batch_target_summaries = target_summaries[i:i + batch_size]
        batch_pmi_generated_summaries = pmi_generated_summaries[i:i + batch_size]
        batch_rouge_generated_summaries = rouge_generated_summaries[i:i + batch_size]

        all_target_summaries.extend(batch_target_summaries)
        all_pmi_predicted_summaries.extend(batch_pmi_generated_summaries)
        all_rouge_predicted_summaries.extend(batch_rouge_generated_summaries)

        # Compute LLM scores for PMI model
        llm_scores_pmi_model = compute_llm_score_batch(batch_pmi_generated_summaries, batch_target_summaries)

        all_pmi_llm_f1_scores.extend(llm_scores_pmi_model)

        # Compute LLM scores for ROUGE model
        llm_scores_rouge_model = compute_llm_score_batch(batch_rouge_generated_summaries, batch_target_summaries)

        all_rouge_llm_f1_scores.extend(llm_scores_rouge_model)

    # Open the combined output file in write mode. Then, add each model's custom LLM scores to the file. For each dictionary in the list stored in the combined results file, add the LLM scores for both models inside the dictionary.
    with open(combined_output_path, "r", encoding="utf-8") as f:
        combined_results = json.load(f)
    for i, result in enumerate(combined_results):
        # Check if the result dictionary has the expected keys, specifically compare ground_truth_summary
        if result["ground_truth_summary"] != all_target_summaries[i]:
            raise ValueError(
                f"Mismatch in ground truth summary at index {i}. Expected: {all_target_summaries[i]}, Found: {result['ground_truth_summary']}")

        result["pmi_pegasus_llm_f1_score"] = all_pmi_llm_f1_scores[i]
        result["rouge_pegasus_llm_f1_score"] = all_rouge_llm_f1_scores[i]

    # Save the updated combined results to a new file
    with open(combined_output_path_with_llm_score, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4)

    # Print a message indicating that the new combined results file that contains new custom LLM scores have been written (cnn dataset)
    print(f"\nThe new custom LLM F1 scores written to {combined_output_path_with_llm_score}")


if __name__ == "__main__":
    # Calculate LLM F1 metric for XSUM dataset
    calc_llm_f1_metric_of_xsum()

    # Calculate LLM F1 metric for CNN dataset
    calc_llm_f1_metric_of_cnn()


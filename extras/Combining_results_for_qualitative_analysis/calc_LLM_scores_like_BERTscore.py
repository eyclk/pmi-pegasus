import torch
from transformers import AutoTokenizer, LlamaModel   #  AutoModel      # LlamaModel
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
import json


model_name = "meta-llama/Llama-2-7b-hf"   # "bert-base-uncased"  # "mistralai/Mistral-7B-v0.3"   # "meta-llama/Llama-2-7b-hf"    # roberta-large
token = "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"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Initialize tokenizer with explicit padding and max_length
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Set max_length according to the model
if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length and tokenizer.model_max_length < 100_000:
    max_length = tokenizer.model_max_length
else:
    # Fallback: set a reasonable default
    if "llama" in model_name.lower():
        max_length = 4096
    elif "bert" in model_name.lower():
        max_length = 512
    elif "mistral" in model_name.lower():
        max_length = 8192
    else:
        max_length = 1024

model = LlamaModel.from_pretrained(model_name, token=token).to(device)


# --- CRITICAL NEW STEP: Resize model's token embeddings if tokenizer size changed ---
# This ensures the model's embedding layer can handle the ID of the new pad token.
if len(tokenizer) > model.config.vocab_size:
    print(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))
    # Update model's config vocab size for consistency if needed, though resize_token_embeddings usually handles it.
    model.config.vocab_size = len(tokenizer)

# Ensure pad_token_id is correctly set in the model's generation config for consistency,
# though it's not strictly necessary for embedding extraction.
if tokenizer.pad_token_id is not None:
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id # Often useful for generation, good practice to set
    model.config.bos_token_id = tokenizer.bos_token_id


# Special tokens to exclude (NEW)
special_token_ids = set(tokenizer.all_special_ids)

model.eval()

batch_size = 2  ### I had to reduce it until 2. Even a single instance of the model takes up 14 GBs of VRAM.  !!!!!


def compute_llm_score_batch(candidates, references):
    """
    candidates: list of str
    references: list of str
    Returns: list of f1 scores (float) for each example
    """
    assert len(candidates) == len(references), "Candidates and references must be the same length"

    # Tokenize with explicit padding and truncation
    cand_inputs = tokenizer(
        candidates,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    ).to(device)
    ref_inputs = tokenizer(
        references,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    ).to(device)

    # Added these lines to print the tokenized lengths of candidates and references
    """for i, seq_ids in enumerate(inputs_cand.input_ids):
        print(f"Candidate {i} tokenized length: {seq_ids.shape[0]}")
    for i, seq_ids in enumerate(inputs_ref.input_ids):
        print(f"Reference {i} tokenized length: {seq_ids.shape[0]}")"""

    with torch.no_grad():
        cand_outputs = model(**cand_inputs).last_hidden_state  # (batch, seq_len, hidden)
        ref_outputs = model(**ref_inputs).last_hidden_state

    f1_scores = []

    for i in range(len(candidates)):
        # Get attention mask and filter out special tokens
        cand_mask = cand_inputs.attention_mask[i].bool()
        ref_mask = ref_inputs.attention_mask[i].bool()

        cand_ids = cand_inputs.input_ids[i]
        ref_ids = ref_inputs.input_ids[i]

        # Create boolean masks to filter out padding and special tokens
        cand_valid = [idx not in special_token_ids for idx in cand_ids[cand_mask]]
        ref_valid = [idx not in special_token_ids for idx in ref_ids[ref_mask]]

        cand_emb = cand_outputs[i][cand_mask][cand_valid]  # (cand_len_valid, hidden)
        ref_emb = ref_outputs[i][ref_mask][ref_valid]      # (ref_len_valid, hidden)

        # Normalize
        cand_emb = F.normalize(cand_emb, p=2, dim=-1)
        ref_emb = F.normalize(ref_emb, p=2, dim=-1)

        # Cosine similarity matrix
        sim_matrix = torch.matmul(cand_emb, ref_emb.T)  # (cand_len, ref_len)

        # Precision: max sim for each candidate token
        precision = sim_matrix.max(dim=1).values.mean()

        # Recall: max sim for each reference token
        recall = sim_matrix.max(dim=0).values.mean()

        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1.item())

        #   print(f"\nCandidate {i} F1 score: {f1.item()}\n")

    return f1_scores


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
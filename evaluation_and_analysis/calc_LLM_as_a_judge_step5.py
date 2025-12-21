import json
from datasets import Dataset
from collections import Counter
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


###############################################################################
# CONFIGURATION
###############################################################################

RM_MODEL_NAME = "allenai/Llama-3.1-Tulu-3-8B-RM"
device = "cuda" if torch.cuda.is_available() else "cpu"

eval_for_FactPEGASUS = False

print("Loading Llama-3.1-Tulu-3-8B-RM...")

tokenizer = AutoTokenizer.from_pretrained(RM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    RM_MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

model.eval()
print("Reward model loaded.\n")


def build_rm_prompt(reference, pmi_summary, rouge_summary):
    return f"""### Reference
{reference}

### Candidate A
{pmi_summary}

### Candidate B
{rouge_summary}

### Question
Which candidate summary is better?

### Answer
"""


###############################################################################
# PAIRWISE JUDGING FUNCTION
###############################################################################

def judge_with_rm(reference, pmi_summary, rouge_summary):
    """
    Uses Llama-3.1-Tulu-3-8B-RM to decide which summary is better.
    Returns: "pmi", "rouge", or "tie"
    """

    prompt = build_rm_prompt(reference, pmi_summary, rouge_summary)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            #  temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip().lower()

    # Robust parsing
    if "a" in decoded or "candidate a" in decoded:
        return "pmi"
    elif "b" in decoded or "candidate b" in decoded:
        return "rouge"
    else:
        return "tie"


###############################################################################
# SHARED EVALUATION FUNCTION FOR ANY DATASET
###############################################################################

def run_llm_judge_evaluation(
    test_dataset_path,
    pmi_path,
    rouge_path,
    combined_input_path,
    combined_output_path,
    factpegasus_path=None
):

    print(f"\n=== Loading dataset: {test_dataset_path} ===")
    ds = Dataset.from_file(test_dataset_path)
    df = ds.to_pandas()
    target_summaries = df["summary"].tolist()

    # Load summaries
    with open(pmi_path, "r", encoding="utf-8") as f:
        pmi_summaries = [x.strip() for x in f.readlines()]
    with open(rouge_path, "r", encoding="utf-8") as f:
        rouge_summaries = [x.strip() for x in f.readlines()]

    if eval_for_FactPEGASUS:
        with open(factpegasus_path, "r", encoding="utf-8") as f:
            fact_summaries = [x.strip() for x in f.readlines()]
    else:
        fact_summaries = None

    if len(pmi_summaries) != len(df):
        raise ValueError("PMI summary count mismatch.")
    if len(rouge_summaries) != len(df):
        raise ValueError("ROUGE summary count mismatch.")
    if eval_for_FactPEGASUS and len(fact_summaries) != len(df):
        raise ValueError("FactPEGASUS summary count mismatch.")

    dataset_name = test_dataset_path.split("_")[0].upper()

    # Pairwise RM judging
    print(f"\nStarting LLM-as-a-Judge RM evaluation for {dataset_name} dataset...\n")
    llm_winners = []

    for i in tqdm(range(len(df))):
        winner = judge_with_rm(
            reference=target_summaries[i],
            pmi_summary=pmi_summaries[i],
            rouge_summary=rouge_summaries[i]
        )
        llm_winners.append(winner)

        """if i % 200 == 0:
            print(f"Processed {i}/{len(df)} samples...")"""

    print(f"\nFinished RM judging for {dataset_name} dataset.\n")

    # Load JSON and insert judge results
    with open(combined_input_path, "r", encoding="utf-8") as f:
        combined_results = json.load(f)

    for i, result in enumerate(combined_results):
        if result["ground_truth_summary"] != target_summaries[i]:
            raise ValueError(f"Ground truth mismatch at index {i}")

        result["llm_judge_winner"] = llm_winners[i]

    # Save
    with open(combined_output_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4)

    print(f"\nWrote judged results for {dataset_name} dataset → {combined_output_path}")

    # Stats
    counts = Counter(llm_winners)
    print(f"\n\n=== LLM-as-a-Judge RM Results for {dataset_name} dataset ===")
    print("PMI wins :", counts["pmi"], f"  (%.4f%%)" % (counts["pmi"] / len(df) * 100))
    print("ROUGE wins:", counts["rouge"], f"  (%.4f%%)" % (counts["rouge"] / len(df) * 100))
    print("Ties      :", counts["tie"], f"  (%.4f%%)" % (counts["tie"] / len(df) * 100))
    print("=================================\n\n")


###############################################################################
# MAIN ENTRY POINT (XSUM + CNN)
###############################################################################

if __name__ == "__main__":

    # ------------------------------- XSUM -----------------------------------
    run_llm_judge_evaluation(
        test_dataset_path="xsum_result_files/test_set_xsum/dataset.arrow",
        pmi_path="xsum_result_files/pmi_pegasus_xsum_generated_summaries/generated_predictions.txt",
        rouge_path="xsum_result_files/rouge_pegasus_xsum_generated_summaries/generated_predictions.txt",
        combined_input_path="xsum_result_files/xsum_combined_results_for_analysis__step4.json",
        combined_output_path="xsum_result_files/xsum_combined_results_for_analysis__step5.json",
    )

    print("\n\n**********************************************************\n\n")

    # ------------------------------- CNN ------------------------------------
    run_llm_judge_evaluation(
        test_dataset_path="cnn_result_files/test_set_cnn/data-00000-of-00001.arrow",
        pmi_path="cnn_result_files/pmi_pegasus_cnn_generated_summaries/generated_predictions.txt",
        rouge_path="cnn_result_files/rouge_pegasus_cnn_generated_summaries/generated_predictions.txt",
        combined_input_path="cnn_result_files/cnn_combined_results_for_analysis__step4.json",
        combined_output_path="cnn_result_files/cnn_combined_results_for_analysis__step5.json",
    )

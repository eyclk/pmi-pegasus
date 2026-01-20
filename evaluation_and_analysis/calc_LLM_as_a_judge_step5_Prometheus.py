import json
import random
from collections import Counter

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.conversation import get_conv_template

###############################################################################
# CONFIGURATION
###############################################################################

MODEL_NAME = "prometheus-eval/prometheus-7b-v2.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.01  # deterministic judging

###############################################################################
# LOAD MODEL
###############################################################################

print("Loading Prometheus (HF) with Mistral conversation template...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto",
)
model.eval()

print("Model loaded successfully.\n")

###############################################################################
# RESULT PARSING
###############################################################################

def parse_prometheus_output(decoded_output: str):
    """
    Splits output into:
      - feedback (before [RESULT])
      - raw_result (A / B / TIE)
    """

    ###  print(f"\n~~~~~~~~~~~~~ Raw model output: {decoded_output}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    if "[RESULT]" not in decoded_output:
        return decoded_output.strip(), "TIE_2"

    split_output = decoded_output.split("[RESULT]")
    feedback = ""
    for output in split_output[:-1]:
        feedback += output

    result_part = split_output[-1]
    result_part = result_part.strip().upper()

    if result_part.startswith("A"):
        result = "A"
    elif result_part.startswith("B"):
        result = "B"
    elif result_part.startswith("TIE"):
        result = "TIE"
    else:
        result = "TIE"

    return feedback.strip(), result

###############################################################################
# PROMETHEUS JUDGE
###############################################################################

def judge_with_prometheus(
    reference_summary: str,
    pmi_summary: str,
    rouge_summary: str,
    sample_idx: int,
) -> str:
    """
    Returns decoded winner: 'pmi', 'rouge', or 'tie'
    """

    # Random swap to test positional impartiality
    swap = random.random() < 0.5

    if swap:
        response_A = pmi_summary
        response_B = rouge_summary
    else:
        response_A = rouge_summary
        response_B = pmi_summary

    conv = get_conv_template("mistral")
    conv.set_system_message(
        "You are a fair and precise evaluation assistant. "
        "You compare two candidate summaries against a reference summary. "
        "Follow the evaluation criteria carefully and be impartial."
    )

    instruction = f"""
TASK DESCRIPTION:
1. You are given a Reference Summary and two Candidate Summaries (A and B).
2. Your task is to evaluate the quality of the two Candidate Summaries based on the Reference Summary using the specified Evaluation Criteria.
3. Write a brief feedback that assess the quality of the two candidate summaries strictly based on the given evaluation criteria, not evaluating in general.
4. After writing the feedback, indicate the better candidate summary, either "A" or "B" or "TIE".
5. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B" or "TIE")"
6. Please do not generate any other opening, closing, and explanations.

EVALUATION CRITERIA:
1. **Faithfulness:** Does the summary avoid adding information not present in the Source?
2. **Coverage:** How well does the summary capture the essential points mentioned in the Reference?
3. **Conciseness:** Is the summary brief without sacrificing key details?
4. **Coherence:** Is the summary easy to read and logically organized?

REFERENCE SUMMARY:
{reference_summary}

CANDIDATE A:
{response_A}

CANDIDATE B:
{response_B}

FEEDBACK: 
""".strip()

## 2. Make comparisons between Candidate A, Candidate B, and the Reference Summary. Instead of examining Candidate A and Candidate B separately, go straight to the point and mention the commonalities and differences between them.

    conv.append_message(conv.roles[0], instruction)

    prompt = conv.get_prompt()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    feedback, raw_result = parse_prometheus_output(decoded_output)

    # Decode winner back to PMI / ROUGE
    if swap:
        decoded_winner = (
            "pmi" if raw_result == "A"
            else "rouge" if raw_result == "B"
            else "tie"
        )
    else:
        decoded_winner = (
            "rouge" if raw_result == "A"
            else "pmi" if raw_result == "B"
            else "tie"
        )

    # =================== DEBUG PRINTS ===================

    """print("\n" + "=" * 80)
    print(f"SAMPLE {sample_idx}")
    print("-" * 80)

    print("FEEDBACK:")
    print(feedback)

    print("\nRAW RESULT (A/B/TIE):")
    print(raw_result)

    print("\nDECODED WINNER (PMI/ROUGE/TIE):")
    print(decoded_winner)

    print("\nPOSITIONAL MAPPING:")
    print(f"A = {'PMI' if swap else 'ROUGE'}")
    print(f"B = {'ROUGE' if swap else 'PMI'}")

    print("=" * 80 + "\n")"""

    return decoded_winner

###############################################################################
# MAIN EVALUATION LOOP
###############################################################################

def run_llm_judge_evaluation(
    test_dataset_path: str,
    pmi_path: str,
    rouge_path: str,
    combined_input_path: str,
    combined_output_path: str,
):

    ds = Dataset.from_file(test_dataset_path)
    df = ds.to_pandas()

    reference_summaries = df["summary"].tolist()

    with open(pmi_path, "r", encoding="utf-8") as f:
        pmi_summaries = [line.strip() for line in f]

    with open(rouge_path, "r", encoding="utf-8") as f:
        rouge_summaries = [line.strip() for line in f]

    winners = []

    for i in tqdm(range(len(reference_summaries))):
        winner = judge_with_prometheus(
            reference_summaries[i],
            pmi_summaries[i],
            rouge_summaries[i],
            i,
        )
        winners.append(winner)

    with open(combined_input_path, "r", encoding="utf-8") as f:
        combined_results = json.load(f)

    for i, entry in enumerate(combined_results):
        entry["llm_judge_winner"] = winners[i]

    with open(combined_output_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4, ensure_ascii=False)

    counts = Counter(winners)

    print("\nFINAL AGGREGATE RESULTS")
    print("----------------------")
    print(f"PMI wins   : {counts['pmi']}")
    print(f"ROUGE wins : {counts['rouge']}")
    print(f"TIES       : {counts['tie']}")
    print("----------------------\n")


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

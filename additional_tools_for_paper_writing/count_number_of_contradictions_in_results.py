import json

results_file_for_analysis = "xsum_combined_results_for_analysis__step5_1M.json"

# Open and read the file with all static metrics results and LLM-as-a-judge results
with open(results_file_for_analysis, 'r', encoding='utf-8') as file:
    results_data = json.load(file)

# These are the relevant fields in the results data: pmi_pegasus_rouge1_score, rouge_pegasus_rouge1_score,
#       pmi_pegasus_rouge2_score, rouge_pegasus_rouge2_score,
#       pmi_pegasus_rougeL_score, rouge_pegasus_rougeL_score,
#       pmi_pegasus_bert_score, rouge_pegasus_bert_score,
#       pmi_pegasus_qaeval_f1_score, rouge_pegasus_qaeval_f1_score,
#       pmi_pegasus_deberta_f1_score, rouge_pegasus_deberta_f1_score,
#       llm_judge_winner.

# The static metrics are all of them except for llm_judge_winner, which is the LLM-as-a-judge result.

# Firstly, count the total number of results where PMI dominates static metrics. This means 5 or more of the static metrics favor PMI over ROUGE.
# Then do the same for ROUGE dominating PMI. This means 5 or more of the static metrics favor ROUGE over PMI.
pmi_dominates_count = 0
rouge_dominates_count = 0


# We also will count the number of contradictions between the static metrics and the LLM-as-a-judge results.
llm_as_judge_PMI__and__ROUGE_dominated_count = 0
llm_as_judge_ROUGE__and__PMI_dominated_count = 0

llm_as_judge_TIE__and__PMI_dominated_count = 0  ## TIE are written as "tie" in the results data
llm_as_judge_TIE__and__ROUGE_dominated_count = 0

for result in results_data:
    pmi_better_count = 0
    rouge_better_count = 0

    # Compare each static metric for this result
    if result['pmi_pegasus_rouge1_score'] > result['rouge_pegasus_rouge1_score']:
        pmi_better_count += 1
    elif result['pmi_pegasus_rouge1_score'] < result['rouge_pegasus_rouge1_score']:
        rouge_better_count += 1

    if result['pmi_pegasus_rouge2_score'] > result['rouge_pegasus_rouge2_score']:
        pmi_better_count += 1
    elif result['pmi_pegasus_rouge2_score'] < result['rouge_pegasus_rouge2_score']:
        rouge_better_count += 1

    if result['pmi_pegasus_rougeL_score'] > result['rouge_pegasus_rougeL_score']:
        pmi_better_count += 1
    elif result['pmi_pegasus_rougeL_score'] < result['rouge_pegasus_rougeL_score']:
        rouge_better_count += 1

    if result['pmi_pegasus_bert_score'] > result['rouge_pegasus_bert_score']:
        pmi_better_count += 1
    elif result['pmi_pegasus_bert_score'] < result['rouge_pegasus_bert_score']:
        rouge_better_count += 1

    if result['pmi_pegasus_qaeval_f1_score'] > result['rouge_pegasus_qaeval_f1_score']:
        pmi_better_count += 1
    elif result['pmi_pegasus_qaeval_f1_score'] < result['rouge_pegasus_qaeval_f1_score']:
        rouge_better_count += 1

    if result['pmi_pegasus_deberta_f1_score'] > result['rouge_pegasus_deberta_f1_score']:
        pmi_better_count += 1
    elif result['pmi_pegasus_deberta_f1_score'] < result['rouge_pegasus_deberta_f1_score']:
        rouge_better_count += 1

    # Check for PMI dominance (5 or more metrics favor PMI)
    if pmi_better_count >= 4:
        pmi_dominates_count += 1

        # Check for contradiction with LLM-as-a-judge
        if result['llm_judge_winner'] == 'rouge':
            llm_as_judge_ROUGE__and__PMI_dominated_count += 1
        elif result['llm_judge_winner'] == 'tie':
            llm_as_judge_TIE__and__PMI_dominated_count += 1

    # Check for ROUGE dominance (5 or more metrics favor ROUGE)
    if rouge_better_count >= 4:
        rouge_dominates_count += 1

        # Check for contradiction with LLM-as-a-judge
        if result['llm_judge_winner'] == 'pmi':
            llm_as_judge_PMI__and__ROUGE_dominated_count += 1
        elif result['llm_judge_winner'] == 'tie':
            llm_as_judge_TIE__and__ROUGE_dominated_count += 1


# Print the results
print(f"\nNumber of results where PMI dominates static metrics: {pmi_dominates_count}\n")
print(f"Number of results where ROUGE dominates static metrics: {rouge_dominates_count}\n")
print("----------------------------------\n")
print(f"Number of contradictions where LLM-as-a-judge favors PMI but ROUGE dominates metrics: {llm_as_judge_PMI__and__ROUGE_dominated_count}\n")
print(f"Number of contradictions where LLM-as-a-judge favors ROUGE but PMI dominates metrics: {llm_as_judge_ROUGE__and__PMI_dominated_count}\n")
print(f"Number of contradictions where LLM-as-a-judge is TIE but PMI dominates metrics: {llm_as_judge_TIE__and__PMI_dominated_count}\n")
print(f"Number of contradictions where LLM-as-a-judge is TIE but ROUGE dominates metrics: {llm_as_judge_TIE__and__ROUGE_dominated_count}\n")


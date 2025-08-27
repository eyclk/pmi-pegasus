import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from datasets import Dataset
from transformers import AutoModel
from tqdm import tqdm
from transformers import logging
# import json

# Set the logging level to ERROR to suppress warnings
logging.set_verbosity_error()

AutoModel.from_pretrained("roberta-large")  # , force_download=True   # Download the model early to avoid errors when importing bert_score

eval_for_FactPEGASUS = False  # Set to False if you want to evaluate only for PMI-Pegasus and ROUGE-Pegasus


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

    # global factPegasus_generated_summaries, factPegasus_generated_predictions_file_path, all_factPegasus_predicted_summaries, all_factPegasus_rouge1_scores, all_factPegasus_rouge2_scores, all_factPegasus_rougeL_scores, all_factPegasus_bert_scores
    batch_size = 32

    test_dataset_path = "xsum_result_files/test_set_xsum/dataset.arrow"
    pmi_generated_predictions_file_path = "xsum_result_files/pmi_pegasus_xsum_generated_summaries/generated_predictions.txt"
    rouge_generated_predictions_file_path = "xsum_result_files/rouge_pegasus_xsum_generated_summaries/generated_predictions.txt"
    if eval_for_FactPEGASUS:
        factPegasus_generated_predictions_file_path = "xsum_result_files/factpegasus_public_xsum_generated_summaries/generated_predictions.txt"
    combined_output_path = "xsum_result_files/xsum_combined_results_for_analysis__step1.json"


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


    all_target_summaries = []
    all_pmi_predicted_summaries = []
    all_rouge_predicted_summaries = []
    all_pmi_rouge1_scores = []
    all_pmi_rouge2_scores = []
    all_pmi_rougeL_scores = []
    all_pmi_bert_scores = []
    all_rouge_rouge1_scores = []
    all_rouge_bert_scores = []
    all_rouge_rouge2_scores = []
    all_rouge_rougeL_scores = []
    if eval_for_FactPEGASUS:
        all_factPegasus_predicted_summaries = []
        all_factPegasus_rouge1_scores = []
        all_factPegasus_rouge2_scores = []
        all_factPegasus_rougeL_scores = []
        all_factPegasus_bert_scores = []

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
            all_pmi_rouge2_scores.append(rouge_scores["rouge2_f1"])
            all_pmi_rougeL_scores.append(rouge_scores["rougeL_f1"])

        # Calculate BERT F1 scores for the PMI generated summaries
        pmi_bert_scores = bert_score(batch_pmi_generated_summaries, batch_target_summaries, lang="en",
                                     model_type="roberta-large", rescale_with_baseline=True)  # rescale_with_baseline=True
        all_pmi_bert_scores.extend(pmi_bert_scores[2].tolist())

        # Calculate ROUGE1 F1 scores for the ROUGE generated summaries
        for target, rouge_summary in zip(batch_target_summaries, batch_rouge_generated_summaries):
            rouge_scores = compute_rouge(target, rouge_summary)
            all_rouge_rouge1_scores.append(rouge_scores["rouge1_f1"])
            all_rouge_rouge2_scores.append(rouge_scores["rouge2_f1"])
            all_rouge_rougeL_scores.append(rouge_scores["rougeL_f1"])

        # Calculate BERT F1 scores for the ROUGE generated summaries
        rouge_bert_scores = bert_score(batch_rouge_generated_summaries, batch_target_summaries, lang="en",
                                       model_type="roberta-large", rescale_with_baseline=True)  # rescale_with_baseline=True
        all_rouge_bert_scores.extend(rouge_bert_scores[2].tolist())

        if eval_for_FactPEGASUS:
            batch_factPegasus_generated_summaries = factPegasus_generated_summaries[i:i + batch_size]
            all_factPegasus_predicted_summaries.extend(batch_factPegasus_generated_summaries)

            # Calculate ROUGE1 F1 scores for the FactPEGASUS generated summaries
            for target, fact_summary in zip(batch_target_summaries, batch_factPegasus_generated_summaries):
                rouge_scores = compute_rouge(target, fact_summary)
                all_factPegasus_rouge1_scores.append(rouge_scores["rouge1_f1"])
                all_factPegasus_rouge2_scores.append(rouge_scores["rouge2_f1"])
                all_factPegasus_rougeL_scores.append(rouge_scores["rougeL_f1"])

            # Calculate BERT F1 scores for the FactPEGASUS generated summaries
            fact_bert_scores = bert_score(batch_factPegasus_generated_summaries, batch_target_summaries, lang="en",
                                         model_type="roberta-large", rescale_with_baseline=True)  # rescale_with_baseline=True
            all_factPegasus_bert_scores.extend(fact_bert_scores[2].tolist())


    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        "ground_truth_summary": all_target_summaries,
        "pmi_pegasus_generated_summary": all_pmi_predicted_summaries,
        "rouge_pegasus_generated_summary": all_rouge_predicted_summaries,
        "pmi_pegasus_rouge1_score": all_pmi_rouge1_scores,
        "rouge_pegasus_rouge1_score": all_rouge_rouge1_scores,
        "pmi_pegasus_rouge2_score": all_pmi_rouge2_scores,
        "rouge_pegasus_rouge2_score": all_rouge_rouge2_scores,
        "pmi_pegasus_rougeL_score": all_pmi_rougeL_scores,
        "rouge_pegasus_rougeL_score": all_rouge_rougeL_scores,
        "pmi_pegasus_bert_score": all_pmi_bert_scores,
        "rouge_pegasus_bert_score": all_rouge_bert_scores
    })

    if eval_for_FactPEGASUS:
        results_df = pd.DataFrame({
            "ground_truth_summary": all_target_summaries,
            "pmi_pegasus_generated_summary": all_pmi_predicted_summaries,
            "rouge_pegasus_generated_summary": all_rouge_predicted_summaries,
            "pmi_pegasus_rouge1_score": all_pmi_rouge1_scores,
            "rouge_pegasus_rouge1_score": all_rouge_rouge1_scores,
            "pmi_pegasus_rouge2_score": all_pmi_rouge2_scores,
            "rouge_pegasus_rouge2_score": all_rouge_rouge2_scores,
            "pmi_pegasus_rougeL_score": all_pmi_rougeL_scores,
            "rouge_pegasus_rougeL_score": all_rouge_rougeL_scores,
            "pmi_pegasus_bert_score": all_pmi_bert_scores,
            "rouge_pegasus_bert_score": all_rouge_bert_scores,
            "factpegasus_generated_summary": all_factPegasus_predicted_summaries,
            "factpegasus_rouge1_score": all_factPegasus_rouge1_scores,
            "factpegasus_rouge2_score": all_factPegasus_rouge2_scores,
            "factpegasus_rougeL_score": all_factPegasus_rougeL_scores,
            "factpegasus_bert_score": all_factPegasus_bert_scores
        })

    # Save the DataFrame to a JSON file with pretty formatting
    results_df.to_json(combined_output_path, orient="records", indent=4)  # , lines=True
    print(f"Combined results saved to {combined_output_path}")

    # Print average scores for both models
    avg_pmi_rouge1 = sum(all_pmi_rouge1_scores) / len(all_pmi_rouge1_scores)
    avg_pmi_rouge2 = sum(all_pmi_rouge2_scores) / len(all_pmi_rouge2_scores)
    avg_pmi_rougeL = sum(all_pmi_rougeL_scores) / len(all_pmi_rougeL_scores)
    avg_pmi_bert = sum(all_pmi_bert_scores) / len(all_pmi_bert_scores)

    avg_rouge_rouge1 = sum(all_rouge_rouge1_scores) / len(all_rouge_rouge1_scores)
    avg_rouge_rouge2 = sum(all_rouge_rouge2_scores) / len(all_rouge_rouge2_scores)
    avg_rouge_rougeL = sum(all_rouge_rougeL_scores) / len(all_rouge_rougeL_scores)
    avg_rouge_bert = sum(all_rouge_bert_scores) / len(all_rouge_bert_scores)

    print(f"\n\nAverage PMI-Pegasus ROUGE1 score for XSUM: {avg_pmi_rouge1:.8f}")
    print(f"Average PMI-Pegasus ROUGE2 score for XSUM: {avg_pmi_rouge2:.8f}")
    print(f"Average PMI-Pegasus ROUGE-L score for XSUM: {avg_pmi_rougeL:.8f}")
    print(f"Average PMI-Pegasus BERT score for XSUM: {avg_pmi_bert:.8f}")

    print(f"\nAverage ROUGE-Pegasus ROUGE1 score for XSUM: {avg_rouge_rouge1:.8f}")
    print(f"Average ROUGE-Pegasus ROUGE2 score for XSUM: {avg_rouge_rouge2:.8f}")
    print(f"Average ROUGE-Pegasus ROUGE-L score for XSUM: {avg_rouge_rougeL:.8f}")
    print(f"Average ROUGE-Pegasus BERT score for XSUM: {avg_rouge_bert:.8f}")

    if eval_for_FactPEGASUS:
        avg_factPegasus_rouge1 = sum(all_factPegasus_rouge1_scores) / len(all_factPegasus_rouge1_scores)
        avg_factPegasus_rouge2 = sum(all_factPegasus_rouge2_scores) / len(all_factPegasus_rouge2_scores)
        avg_factPegasus_rougeL = sum(all_factPegasus_rougeL_scores) / len(all_factPegasus_rougeL_scores)
        avg_factPegasus_bert = sum(all_factPegasus_bert_scores) / len(all_factPegasus_bert_scores)

        print(f"\n\nAverage FactPEGASUS ROUGE1 score for XSUM: {avg_factPegasus_rouge1:.8f}")
        print(f"Average FactPEGASUS ROUGE2 score for XSUM: {avg_factPegasus_rouge2:.8f}")
        print(f"Average FactPEGASUS ROUGE-L score for XSUM: {avg_factPegasus_rougeL:.8f}")
        print(f"Average FactPEGASUS BERT score for XSUM: {avg_factPegasus_bert:.8f}")


def combine_results_of_cnn():

    batch_size = 32

    test_dataset_path = "cnn_result_files/test_set_cnn/data-00000-of-00001.arrow"
    pmi_generated_predictions_file_path = "cnn_result_files/pmi_pegasus_cnn_generated_summaries/generated_predictions.txt"
    rouge_generated_predictions_file_path = "cnn_result_files/rouge_pegasus_cnn_generated_summaries/generated_predictions.txt"

    if eval_for_FactPEGASUS:
        factPegasus_generated_predictions_file_path = "cnn_result_files/factpegasus_public_cnn_generated_summaries/generated_predictions.txt"

    combined_output_path = "cnn_result_files/cnn_combined_results_for_analysis__step1.json"


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


    all_target_summaries = []
    all_pmi_predicted_summaries = []
    all_rouge_predicted_summaries = []
    all_pmi_rouge1_scores = []
    all_pmi_bert_scores = []
    all_rouge_rouge1_scores = []
    all_rouge_bert_scores = []
    all_pmi_rouge2_scores = []
    all_pmi_rougeL_scores = []
    all_rouge_rouge2_scores = []
    all_rouge_rougeL_scores = []
    if eval_for_FactPEGASUS:
        all_factPegasus_predicted_summaries = []
        all_factPegasus_rouge1_scores = []
        all_factPegasus_rouge2_scores = []
        all_factPegasus_rougeL_scores = []
        all_factPegasus_bert_scores = []

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
            all_pmi_rouge2_scores.append(rouge_scores["rouge2_f1"])
            all_pmi_rougeL_scores.append(rouge_scores["rougeL_f1"])

        # Calculate BERT F1 scores for the PMI generated summaries
        pmi_bert_scores = bert_score(batch_pmi_generated_summaries, batch_target_summaries, lang="en",
                                     model_type="roberta-large", rescale_with_baseline=True)  # rescale_with_baseline=True
        all_pmi_bert_scores.extend(pmi_bert_scores[2].tolist())

        # Calculate ROUGE1 F1 scores for the ROUGE generated summaries
        for target, rouge_summary in zip(batch_target_summaries, batch_rouge_generated_summaries):
            rouge_scores = compute_rouge(target, rouge_summary)
            all_rouge_rouge1_scores.append(rouge_scores["rouge1_f1"])
            all_rouge_rouge2_scores.append(rouge_scores["rouge2_f1"])
            all_rouge_rougeL_scores.append(rouge_scores["rougeL_f1"])

        # Calculate BERT F1 scores for the ROUGE generated summaries
        rouge_bert_scores = bert_score(batch_rouge_generated_summaries, batch_target_summaries, lang="en",
                                       model_type="roberta-large", rescale_with_baseline=True)  # rescale_with_baseline=True
        all_rouge_bert_scores.extend(rouge_bert_scores[2].tolist())

        if eval_for_FactPEGASUS:
            batch_factPegasus_generated_summaries = factPegasus_generated_summaries[i:i + batch_size]
            all_factPegasus_predicted_summaries.extend(batch_factPegasus_generated_summaries)

            # Calculate ROUGE1 F1 scores for the FactPEGASUS generated summaries
            for target, fact_summary in zip(batch_target_summaries, batch_factPegasus_generated_summaries):
                rouge_scores = compute_rouge(target, fact_summary)
                all_factPegasus_rouge1_scores.append(rouge_scores["rouge1_f1"])
                all_factPegasus_rouge2_scores.append(rouge_scores["rouge2_f1"])
                all_factPegasus_rougeL_scores.append(rouge_scores["rougeL_f1"])

            # Calculate BERT F1 scores for the FactPEGASUS generated summaries
            fact_bert_scores = bert_score(batch_factPegasus_generated_summaries, batch_target_summaries, lang="en",
                                         model_type="roberta-large", rescale_with_baseline=True)  # rescale_with_baseline=True
            all_factPegasus_bert_scores.extend(fact_bert_scores[2].tolist())


    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        "ground_truth_summary": all_target_summaries,
        "pmi_pegasus_generated_summary": all_pmi_predicted_summaries,
        "rouge_pegasus_generated_summary": all_rouge_predicted_summaries,
        "pmi_pegasus_rouge1_score": all_pmi_rouge1_scores,
        "rouge_pegasus_rouge1_score": all_rouge_rouge1_scores,
        "pmi_pegasus_rouge2_score": all_pmi_rouge2_scores,
        "rouge_pegasus_rouge2_score": all_rouge_rouge2_scores,
        "pmi_pegasus_rougeL_score": all_pmi_rougeL_scores,
        "rouge_pegasus_rougeL_score": all_rouge_rougeL_scores,
        "pmi_pegasus_bert_score": all_pmi_bert_scores,
        "rouge_pegasus_bert_score": all_rouge_bert_scores
    })

    if eval_for_FactPEGASUS:
        results_df = pd.DataFrame({
            "ground_truth_summary": all_target_summaries,
            "pmi_pegasus_generated_summary": all_pmi_predicted_summaries,
            "rouge_pegasus_generated_summary": all_rouge_predicted_summaries,
            "pmi_pegasus_rouge1_score": all_pmi_rouge1_scores,
            "rouge_pegasus_rouge1_score": all_rouge_rouge1_scores,
            "pmi_pegasus_rouge2_score": all_pmi_rouge2_scores,
            "rouge_pegasus_rouge2_score": all_rouge_rouge2_scores,
            "pmi_pegasus_rougeL_score": all_pmi_rougeL_scores,
            "rouge_pegasus_rougeL_score": all_rouge_rougeL_scores,
            "pmi_pegasus_bert_score": all_pmi_bert_scores,
            "rouge_pegasus_bert_score": all_rouge_bert_scores,
            "factpegasus_generated_summary": all_factPegasus_predicted_summaries,
            "factpegasus_rouge1_score": all_factPegasus_rouge1_scores,
            "factpegasus_rouge2_score": all_factPegasus_rouge2_scores,
            "factpegasus_rougeL_score": all_factPegasus_rougeL_scores,
            "factpegasus_bert_score": all_factPegasus_bert_scores
        })

    # Save the DataFrame to a JSON file with pretty formatting
    results_df.to_json(combined_output_path, orient="records", indent=4)  # , lines=True
    print(f"Combined results saved to {combined_output_path}")

    # Print average scores for both models
    avg_pmi_rouge1 = sum(all_pmi_rouge1_scores) / len(all_pmi_rouge1_scores)
    avg_pmi_rouge2 = sum(all_pmi_rouge2_scores) / len(all_pmi_rouge2_scores)
    avg_pmi_rougeL = sum(all_pmi_rougeL_scores) / len(all_pmi_rougeL_scores)
    avg_pmi_bert = sum(all_pmi_bert_scores) / len(all_pmi_bert_scores)

    avg_rouge_rouge1 = sum(all_rouge_rouge1_scores) / len(all_rouge_rouge1_scores)
    avg_rouge_rouge2 = sum(all_rouge_rouge2_scores) / len(all_rouge_rouge2_scores)
    avg_rouge_rougeL = sum(all_rouge_rougeL_scores) / len(all_rouge_rougeL_scores)
    avg_rouge_bert = sum(all_rouge_bert_scores) / len(all_rouge_bert_scores)

    print(f"\n\nAverage PMI-Pegasus ROUGE1 score for CNN: {avg_pmi_rouge1:.8f}")
    print(f"Average PMI-Pegasus ROUGE2 score for CNN: {avg_pmi_rouge2:.8f}")
    print(f"Average PMI-Pegasus ROUGE-L score for CNN: {avg_pmi_rougeL:.8f}")
    print(f"Average PMI-Pegasus BERT score for CNN: {avg_pmi_bert:.8f}")

    print(f"\nAverage ROUGE-Pegasus ROUGE1 score for CNN: {avg_rouge_rouge1:.8f}")
    print(f"Average ROUGE-Pegasus ROUGE2 score for CNN: {avg_rouge_rouge2:.8f}")
    print(f"Average ROUGE-Pegasus ROUGE-L score for CNN: {avg_rouge_rougeL:.8f}")
    print(f"Average ROUGE-Pegasus BERT score for CNN: {avg_rouge_bert:.8f}")

    if eval_for_FactPEGASUS:
        avg_factPegasus_rouge1 = sum(all_factPegasus_rouge1_scores) / len(all_factPegasus_rouge1_scores)
        avg_factPegasus_rouge2 = sum(all_factPegasus_rouge2_scores) / len(all_factPegasus_rouge2_scores)
        avg_factPegasus_rougeL = sum(all_factPegasus_rougeL_scores) / len(all_factPegasus_rougeL_scores)
        avg_factPegasus_bert = sum(all_factPegasus_bert_scores) / len(all_factPegasus_bert_scores)

        print(f"\n\nAverage FactPEGASUS ROUGE1 score for CNN: {avg_factPegasus_rouge1:.8f}")
        print(f"Average FactPEGASUS ROUGE2 score for CNN: {avg_factPegasus_rouge2:.8f}")
        print(f"Average FactPEGASUS ROUGE-L score for CNN: {avg_factPegasus_rougeL:.8f}")
        print(f"Average FactPEGASUS BERT score for CNN: {avg_factPegasus_bert:.8f}")


### ONLY USED FOR BASIC TEST WITH CHATGPT
"""def reformat_combined_results_for_llm_models(path_to_combined_results):

    # Load the combined results file from the specified path using json module
    with open(path_to_combined_results, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create a DataFrame from the loaded data
    df = pd.DataFrame(data)

    # Rename pmi_pegasus_generated_summary column to model_A_generated_summary
    df.rename(columns={"pmi_pegasus_generated_summary": "model_A_generated_summary"}, inplace=True)

    # Rename rouge_pegasus_generated_summary column to model_B_generated_summary
    df.rename(columns={"rouge_pegasus_generated_summary": "model_B_generated_summary"}, inplace=True)

    # Remove all score columns
    df.drop(columns=["pmi_pegasus_rouge1_score", "rouge_pegasus_rouge1_score", "pmi_pegasus_bert_score", "rouge_pegasus_bert_score"], inplace=True)

    # Add two columns named "reasoning_behind_the_comparison" and "comparison_result"
    df["reasoning_behind_the_comparison"] = ""
    df["comparison_result"] = ""

    # Save the reformatted DataFrame to a new JSON file
    reformatted_output_path = path_to_combined_results.replace(".json", "_reformatted.json")
    df.to_json(reformatted_output_path, orient="records", indent=4)  #  , lines=True
    print(f"Reformatted results saved to {reformatted_output_path}")"""


if __name__ == "__main__":

    combine_results_of_xsum()

    print("\n\n**********************************************************\n\n")

    combine_results_of_cnn()

import pandas as pd
import json


def count_better_model_instances(dataset_name="xsum", metric_name="rouge1", complete_file_name=""):
    result_file_path = f"./{dataset_name}_result_files/{dataset_name}_combined_results_for_analysis{complete_file_name}.json"

    # Load the combined results into a DataFrame
    with open(result_file_path, 'r') as f:
        combined_results = json.load(f)
    df = pd.DataFrame(combined_results)

    better_pmi_count = 0
    better_rouge_count = 0
    equal_count = 0

    pmi_metric_name_in_df = f"pmi_pegasus_{metric_name}_score"
    rouge_metric_name_in_df = f"rouge_pegasus_{metric_name}_score"

    # Iterate through the DataFrame and count the instances
    for index, row in df.iterrows():
        if row[pmi_metric_name_in_df] > row[rouge_metric_name_in_df]:
            better_pmi_count += 1
        elif row[pmi_metric_name_in_df] < row[rouge_metric_name_in_df]:
            better_rouge_count += 1
        else:
            equal_count += 1


    print(f"\nNumber of instances where PMI-Pegasus Model is better in {dataset_name} dataset for {metric_name} metric: {better_pmi_count}")
    print(f"\nNumber of instances where ROUGE-Pegasus Model is better in {dataset_name} dataset for {metric_name} metric: {better_rouge_count}")
    print(f"\nNumber of instances where both models are equal in {dataset_name} dataset for {metric_name} metric: {equal_count}\n\n")


def count_instances_where_both_metrics_agree(dataset_name="xsum"):
    result_file_path = f"./{dataset_name}_result_files/{dataset_name}_combined_results_for_analysis.json"

    # Load the combined results into a DataFrame
    with open(result_file_path, 'r') as f:
        combined_results = json.load(f)
    df = pd.DataFrame(combined_results)

    both_agree_for_pmi_pegasus_count = 0
    both_agree_for_rouge_pegasus_count = 0

    pmi_rouge1_in_df = f"pmi_pegasus_rouge1_score"
    rouge_rouge1_in_df = f"rouge_pegasus_rouge1_score"
    pmi_bert_in_df = f"pmi_pegasus_bert_score"
    rouge_bert_in_df = f"rouge_pegasus_bert_score"

    # Iterate through the DataFrame and count the instances where pmi_rouge1_in_df > rouge_rouge1_in_df and pmi_bert_in_df > rouge_bert_in_df
    # Also, do this where pmi_rouge1_in_df < rouge_rouge1_in_df and pmi_bert_in_df < rouge_bert_in_df
    for index, row in df.iterrows():
        if row[pmi_rouge1_in_df] > row[rouge_rouge1_in_df] and row[pmi_bert_in_df] > row[rouge_bert_in_df]:
            both_agree_for_pmi_pegasus_count += 1
        elif row[pmi_rouge1_in_df] < row[rouge_rouge1_in_df] and row[pmi_bert_in_df] < row[rouge_bert_in_df]:
            both_agree_for_rouge_pegasus_count += 1

    print(f"\nNumber of instances where both metrics agree that PMI-Pegasus is better in {dataset_name} dataset: {both_agree_for_pmi_pegasus_count}")
    print(f"\nNumber of instances where both metrics agree that ROUGE-Pegasus is better in {dataset_name} dataset: {both_agree_for_rouge_pegasus_count}\n\n")


def paired_t_test_of_both_scores_of_a_model(dataset_name="xsum", metric_name="rouge1", complete_file_name=""):
    result_file_path = f"./{dataset_name}_result_files/{dataset_name}_combined_results_for_analysis{complete_file_name}.json"
    ###### Use this name for the previous version:  "{dataset_name}_result_files__1_mil_subset__old"

    # Load the combined results into a DataFrame
    with open(result_file_path, 'r') as f:
        combined_results = json.load(f)
    df = pd.DataFrame(combined_results)

    pmi_metric_name_in_df = f"pmi_pegasus_{metric_name}_score"
    rouge_metric_name_in_df = f"rouge_pegasus_{metric_name}_score"


    # Remove all 0 values from df[pmi_metric_name_in_df] and df[rouge_metric_name_in_df] only when both are 0
    # ----------------->>> NEW !!
    pmi_metric_values_without_zeros = df[pmi_metric_name_in_df][(df[pmi_metric_name_in_df] != 0) | (df[rouge_metric_name_in_df] != 0)].values
    rouge_metric_values_without_zeros = df[rouge_metric_name_in_df][(df[pmi_metric_name_in_df] != 0) | (df[rouge_metric_name_in_df] != 0)].values
    #  print(pmi_metric_values_without_zeros[:10])
    # ----------------->>> NEW !!


    # Perform paired t-test
    from scipy import stats
    t_statistic, p_value = stats.ttest_rel(pmi_metric_values_without_zeros, rouge_metric_values_without_zeros)

    print(f"\nPaired t-test statistic for <{dataset_name}> dataset for \'{metric_name}\' metric between both models: {t_statistic}")
    print(f"\nPaired t-test p-value for <{dataset_name}> dataset for \'{metric_name}\' metric between both models: {p_value}\n")

    # Interpretation
    if p_value < 0.05:
        print(f"--> There is \"a statistically significant difference\" between the metrics of models for <{dataset_name}> dataset.\n")
    else:
        print(f"--> There is \"NO statistically significant difference\" between the metrics of models for <{dataset_name}> dataset.\n")

    # Interpret t-statistic: If it is negative, it means the first metric (PMI) is lower than the second metric (ROUGE), and vice versa.
    if t_statistic < 0:
        print(f"--> The first model (PMI) mean is LOWER than the second model (ROUGE) for <{dataset_name}> dataset.\n\n")
    else:
        print(f"--> The first model (PMI) mean is HIGHER than the second model (ROUGE) for <{dataset_name}> dataset.\n\n")


if __name__ == "__main__":

    """count_better_model_instances(dataset_name="xsum", metric_name="rouge1", complete_file_name="")

    count_better_model_instances(dataset_name="xsum", metric_name="bert", complete_file_name="")

    count_better_model_instances(dataset_name="cnn", metric_name="rouge1", complete_file_name="")

    count_better_model_instances(dataset_name="cnn", metric_name="bert", complete_file_name="")

    count_better_model_instances(dataset_name="xsum", metric_name="qaeval_f1", complete_file_name="__with_qaeval")

    count_better_model_instances(dataset_name="cnn", metric_name="qaeval_f1", complete_file_name="__with_qaeval")

    count_better_model_instances(dataset_name="xsum", metric_name="qaeval_is_answered", complete_file_name="__with_qaeval")

    count_better_model_instances(dataset_name="cnn", metric_name="qaeval_is_answered", complete_file_name="__with_qaeval")

    count_better_model_instances(dataset_name="xsum", metric_name="deberta_f1",
                                 complete_file_name="__with_deberta_score")

    count_better_model_instances(dataset_name="cnn", metric_name="deberta_f1",
                                 complete_file_name="__with_deberta_score")

    count_better_model_instances(dataset_name="xsum", metric_name="llama_f1",
                                 complete_file_name="__with_llama_score")

    count_better_model_instances(dataset_name="cnn", metric_name="llama_f1",
                                 complete_file_name="__with_llama_score")"""


    # count_instances_where_both_metrics_agree(dataset_name="xsum")

    # count_instances_where_both_metrics_agree(dataset_name="cnn")


    paired_t_test_of_both_scores_of_a_model(dataset_name="xsum", metric_name="rouge1", complete_file_name="__step1")

    paired_t_test_of_both_scores_of_a_model(dataset_name="xsum", metric_name="rouge2", complete_file_name="__step1")

    paired_t_test_of_both_scores_of_a_model(dataset_name="xsum", metric_name="rougeL", complete_file_name="__step1")

    paired_t_test_of_both_scores_of_a_model(dataset_name="xsum", metric_name="bert", complete_file_name="__step1")

    paired_t_test_of_both_scores_of_a_model(dataset_name="cnn", metric_name="rouge1", complete_file_name="__step1")

    paired_t_test_of_both_scores_of_a_model(dataset_name="cnn", metric_name="rouge2", complete_file_name="__step1")

    paired_t_test_of_both_scores_of_a_model(dataset_name="cnn", metric_name="rougeL", complete_file_name="__step1")

    paired_t_test_of_both_scores_of_a_model(dataset_name="cnn", metric_name="bert", complete_file_name="__step1")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    paired_t_test_of_both_scores_of_a_model(dataset_name="xsum", metric_name="qaeval_f1", complete_file_name="__step2")

    paired_t_test_of_both_scores_of_a_model(dataset_name="cnn", metric_name="qaeval_f1", complete_file_name="__step2")

    paired_t_test_of_both_scores_of_a_model(dataset_name="xsum", metric_name="qaeval_is_answered", complete_file_name="__step2")

    paired_t_test_of_both_scores_of_a_model(dataset_name="cnn", metric_name="qaeval_is_answered", complete_file_name="__step2")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    paired_t_test_of_both_scores_of_a_model(dataset_name="xsum", metric_name="deberta_f1",
                                            complete_file_name="__step3")

    paired_t_test_of_both_scores_of_a_model(dataset_name="cnn", metric_name="deberta_f1",
                                            complete_file_name="__step3")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    paired_t_test_of_both_scores_of_a_model(dataset_name="xsum", metric_name="llama_f1",
                                            complete_file_name="__step4")

    paired_t_test_of_both_scores_of_a_model(dataset_name="cnn", metric_name="llama_f1",
                                            complete_file_name="__step4")


# PMI-PEGASUS

This repository contains the implementation and experimental setup for **PMI-PEGASUS** and **ROUGE-PEGASUS**.

This guide explains how to reproduce:

* dataset preprocessing
* pretraining and fine-tuning
* evaluation with ROUGE, BERTScore, QAeval, and LLM-as-Judge

##  Environment Overview

| Environment          | Purpose                         |
| -------------------- | ------------------------------- |
| `pegasus_preprocess` | C4 dataset preprocessing        |
| `pegasus_pretrain`   | Pretraining and fine-tuning     |
| `pmi_pegasus`        | Blackwell GPU compatibility     |
| `ft_data_prepare`    | Fine-tuning dataset preparation |
| `pegasus_eval`       | ROUGE and BERTScore evaluation  |
| `pegasus_qaeval`     | QA-based evaluation             |
| `prometheus`         | LLM-as-Judge evaluation         |

##  1. Preprocessing Environment

This environment is used for preprocessing the **C4 realnewslike subset**.

> PMI preprocessing is computationally expensive. It is recommended to process data in chunks, such as 1 million samples at a time, and merge them afterward.

It is also recommended to use `transformers==4.10.0` for preprocessing, as it improves preprocessing speed.

### Setup

```bash
conda create -n pegasus_preprocess python=3.9
conda activate pegasus_preprocess

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install datasets==2.0.0
pip install transformers==4.10.0
pip install nltk==3.7
pip install numpy==1.26.4
```

### Preprocessing Commands

Due to an artifact inherited from the original codebase, preprocessing must be run in two steps for each approach.

```bash
python scripts/pretraining_create_data_for_PMI.py
python scripts/pretraining_combine_scores_for_PMI.py

python scripts/pretraining_create_data_for_Rouge.py
python scripts/pretraining_combine_scores_for_Rouge.py
```

### Additional Training Dataset Preprocessing Notes

You can modify the parameters at the of the pretraining_create_data and pretraining_combine_data files. It is possible to change input/output paths and partial dataset preprocessing details.


##  2. Pretraining and Fine-Tuning Environment

This environment is used for both **pretraining** and **fine-tuning** with the PMI and ROUGE approaches.

### Setup

```bash
conda create -n pegasus_pretrain python=3.9
conda activate pegasus_pretrain

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install datasets==2.0.0
pip install transformers==4.17.0
pip install deepspeed==0.6.4
pip install nltk==3.7
pip install rouge_score==0.0.4
pip install numpy==1.26.4
pip install tokenizers==0.14.1
pip install sentencepiece
pip install aiohttp==3.11.11
pip install protobuf==3.19.6
pip install pyarrow==19.0.0
```

##  Blackwell GPU Compatibility

If you are using an RTX 5080 or another Blackwell-generation GPU, use the following environment instead.

### Setup

```bash
conda create -n pmi_pegasus python=3.10
conda activate pmi_pegasus

pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install datasets==2.0.0
pip install transformers==4.17.0
pip install deepspeed==0.6.4
pip install nltk==3.7
pip install rouge_score==0.0.4
pip install numpy==1.26.4
pip install tokenizers==0.14.1
pip install sentencepiece
pip install aiohttp==3.11.11
pip install protobuf==3.19.6
pip install pyarrow==19.0.0
```

### Additional Fixes for Blackwell GPUs

Some DeepSpeed files may require a manual compatibility fix.

Open the following files:

```bash
nano /home/ege/miniconda3/envs/pmi_pegasus/lib/python3.10/site-packages/deepspeed/runtime/utils.py
nano /home/ege/miniconda3/envs/pmi_pegasus/lib/python3.10/site-packages/deepspeed/runtime/zero/stage_1_and_2.py
```

Replace:

```python
from torch._six import inf
```

with:

```python
try:
    from torch._six import inf
except ImportError:
    from torch import inf
```

Also downgrade NumPy if necessary:

```bash
pip install "numpy<2.0"
```

To reduce GPU memory pressure during fine-tuning, use the following safer settings:

```bash
per_device_train_batch_size=8
gradient_accumulation_steps=4
```

instead of:

```bash
per_device_train_batch_size=16
gradient_accumulation_steps=2
```

##  Training Configuration Notes

Before running pretraining or fine-tuning, review the relevant shell scripts and adjust the following parameters if needed.

### GPU Selection

If necessary, set the GPU index inside the `.sh` file:

```bash
GPU_IDX=0
```

### Dataset Path

Set the `data_dir` parameter inside the shell script to the location of the combined preprocessing output. For example:

```bash
data_dir="./c4_realnewslike_processed_PMI_combined"
```

### Batch Size

If your GPU has enough VRAM, you may increase:

```bash
per_device_train_batch_size=16
```

Otherwise, reduce it as needed.

### Quick Trial Runs

For quick sanity checks or early experiments, you can reduce the number of steps. For example:

```bash
max_steps=500
```

##  Recommended Directory Structure

Create the following directories inside the project folder:

```bash
mkdir models
mkdir finetuned_models
mkdir preprocessed_pretrain_datasets
mkdir preprocessed_finetune_datasets
```

* `models/` stores pretrained checkpoints
* `finetuned_models/` stores fine-tuned checkpoints
* additional folders for preprocessed datasets are optional but recommended for organization

##  Running Pretraining

Use the following commands to run pretraining for each approach:

```bash
./run_pretrain_pegasus_PMI.sh
./run_pretrain_pegasus_ROUGE.sh
```

To continue PMI pretraining from a checkpoint:

```bash
./run_pretrain_pegasus_PMI-from_a_checkpoint.sh
```

##  Running Fine-Tuning

Use the following commands to fine-tune pretrained models:

```bash
./finetune_PMI_pegasus.sh
./finetune_ROUGE_pegasus.sh
```

##  Generating Summaries After Fine-Tuning

After fine-tuning, run the following script to generate summaries for each test-set example:

```bash
./eval_finetuned_models.sh
```

The resulting text outputs can then be used with the evaluation scripts inside the `evaluation_and_analysis` folder.

## 3. Fine-Tuning Dataset Preparation

This environment is used for preparing fine-tuning datasets such as **CNN/DailyMail** and **XSUM** with scripts such as:

* `create_dataset.py`
* `run_spacy.py`
* `corrector.py`

Alternative fixed versions are also available:

* `run_spacy_for_cnn.py`
* `corrector_no_shards.py`

This environment uses the same requirements as the preprocessing environment, plus the additional libraries required for spaCy.

### Setup

```bash
conda create -n ft_data_prepare python=3.9
conda activate ft_data_prepare

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install datasets==2.0.0
pip install transformers==4.10.0
pip install nltk==3.7
pip install numpy==1.26.4
pip install spacy
```

### Example: Preparing the XSUM Dataset

```bash
python scripts/create_dataset.py xsum
python scripts/run_spacy.py xsum
python scripts/corrector.py --data_dir data/xsum_tokens --save_dir data/xsum_comb --correction_type all
```

Valid correction types are:

* `all`
* `remove`
* `replace`

## 4. Evaluation with ROUGE and BERTScore

This environment is used for evaluating summaries with **ROUGE** and **BERTScore**, including settings based on models such as RoBERTa-large and DeBERTa.

### Setup

```bash
conda create -n pegasus_eval python=3.9
conda activate pegasus_eval

pip install torch
pip install datasets
pip install transformers
pip install scikit-learn pandas
pip install bert-score
pip install rouge-score
```

Use the most recent compatible Torch and Transformers versions in this environment.

## 5. Evaluation with QAeval

This environment is used for **QAeval**, including both **F1** and **is_answered** scores.

### Setup

```bash
conda create -n pegasus_qaeval python=3.9
conda activate pegasus_qaeval

pip install datasets
pip install -U spacy
python -m spacy download en_core_web_sm

conda install python==3.8

pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install sacrerouge==0.2.2
pip install qaeval==0.0.9
pip uninstall googledrivedownloader -y
pip install googledrivedownloader==0.4

sacrerouge setup-metric qa-eval
```

### Important Notes

The `sacrerouge setup-metric qa-eval` command may fail. In that case, the QAeval model files must be downloaded manually and copied into the folders created by that command.

You may also need to edit the following file:

```bash
nano /home/audp/anaconda3/envs/qaeval/lib/python3.8/site-packages/datasets/utils/_dill.py
```

In that file, replace:

```python
spacy.Language
```

with:

```python
spacy.language.Language
```

If you need to manually download QAeval models, you can use the commands listed in `download_qaeval_models.txt`.

### Optional GPU Support for QAeval

To run QAeval with GPU acceleration, install:

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

The code can also be updated to use:

```python
qa_metric = QAEval(cuda_device=0)
```

This may produce warnings, but it can run faster.

## 6. Evaluation with Prometheus (LLM-as-Judge)

This environment is used for **Prometheus-based LLM-as-Judge evaluation**.

### Setup

```bash
conda create -n prometheus python=3.12
conda activate prometheus

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install datasets
pip install transformers
pip install accelerate
pip install fastchat
pip install sentencepiece
```

## Evaluation Notes

You can run evaluation step files inside the "evaluation_and_analysis" directory one by one to acquire result JSON files that contain very detailed outputs for all metrics and datasets.

## 📝 Reproducibility Notes

* All experiments assume the **C4 realnewslike subset**
* PMI preprocessing is significantly slower than the ROUGE-based preprocessing pipeline
* Chunked preprocessing is strongly recommended for PMI
* Use separate environments for preprocessing, training, and evaluation to avoid version conflicts
* Keep library versions consistent if you want to reproduce the original results as closely as possible

## 7. SBERT-PEGASUS (Embedding-Based Principal Sentence Selection)

SBERT-PEGASUS is a third principal sentence selection method, added next to PMI and ROUGE. Instead of the lexical overlap (ROUGE) or pointwise mutual information (PMI) between a sentence and the rest of its document, each sentence is scored by the **cosine similarity between its SBERT embedding and the embedding of the rest of the document**. The highest-scoring sentence is masked out and becomes the pretraining target.

All SBERT-specific preprocessing code lives in `scripts/extra_preprocessing_codes_for_SBERT/`.

### Preprocessing

```bash
conda activate pegasus_preprocess
pip install huggingface_hub   # if not already installed

python scripts/extra_preprocessing_codes_for_SBERT/pretraining_create_data_for_SBERT.py
```

Unlike PMI and ROUGE, there is **no separate combine step**: the script writes the final set directly to:

```
./PREPROCESSED_DATASETS/c4_realnewslike_processed_SBERT_complete_combined
```

Useful options:

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--source` | `preprocessed` | `preprocessed` rebuilds the source texts from an existing preprocessed set (no C4 re-download, and the SBERT set stays aligned example by example with it). `c4` loads C4 directly. |
| `--source_dataset` | `./PREPROCESSED_DATASETS/c4_realnewslike_processed_ROUGE_complete_combined` | The preprocessed set used when `--source preprocessed`. |
| `--sbert_model` | `sentence-transformers/all-MiniLM-L6-v2` | Hub name or a local directory. If the machine cannot reach the Hugging Face hub, download the model elsewhere and pass its folder. |
| `--doc_repr` | `mean_of_others` | How the "rest of the document" is represented: mean of the other sentence embeddings (fast), or `leave_one_out_text` (re-encodes the remaining text). |
| `--map_cache_dir` | `./PREPROCESSED_DATASETS/sbert_map_cache` | Temporary cache for the final map; deleted after saving. Needs roughly one extra copy of the dataset in disk space. |

Batch sizes, worker counts, FP16 and partial-subset processing (`USE_SMALLER_SUBSET`, `SUBSET_LOWER_LIMIT`, `SUBSET_UPPER_LIMIT`) are set as constants at the top of the script.

### Comparing Principal Sentence Selections

These scripts measure how often two or three methods pick the same principal sentence. The compared sets must be built from the same slice of C4, so that the same index refers to the same source document (alignment is verified by default).

```bash
# Two sets
python scripts/extra_preprocessing_codes_for_SBERT/compare_principal_sent_of_two_preprocessed_sets.py \
    --dataset_a ./PREPROCESSED_DATASETS/c4_realnewslike_processed_ROUGE_complete_combined \
    --dataset_b ./PREPROCESSED_DATASETS/c4_realnewslike_processed_SBERT_complete_combined \
    --name_a ROUGE --name_b SBERT

# PMI vs ROUGE vs a third set
python scripts/extra_preprocessing_codes_for_SBERT/compare_principal_sent_of_three_preprocessed_sets.py \
    --dataset_pmi ./PREPROCESSED_DATASETS/c4_realnewslike_processed_PMI_complete_combined \
    --dataset_rouge ./PREPROCESSED_DATASETS/c4_realnewslike_processed_ROUGE_complete_combined \
    --dataset_other ./PREPROCESSED_DATASETS/c4_realnewslike_processed_SBERT_complete_combined \
    --name_other SBERT --print_first_k_differences 10
```

Use `--max_examples N` for a quick check on the first N examples. The existing `diff_*_vs_SBERT__output.txt` files in the same folder are the outputs of these comparisons.

### Pretraining and Fine-Tuning

SBERT-PEGASUS uses the same `pegasus_pretrain` (or `pmi_pegasus`) environment and the same training code as PMI and ROUGE:

```bash
./run_pretrain_pegasus_SBERT.sh
./finetune_SBERT_pegasus.sh
```

As with the other scripts, edit `GPU_IDX`, `--model_name`, `--data_dir`, `--max_target_length` and `--output_dir` inside the files before running. For a fair comparison with PMI/ROUGE, use `--max_target_length 64` for XSUM and `128` for CNN/DailyMail and WikiHow.

### Automated End-to-End Pipeline

`run_SBERT_full_pipeline_4M_to_8M.sh` runs the whole SBERT workflow without manual steps. It skips stages whose outputs already exist, so it can simply be re-run after a crash, and it writes logs under `./pipeline_logs/`.

For each checkpoint (5M, 6M, 7M, 8M by default), it:

1. resumes pretraining from the previous checkpoint for 1M more steps,
2. fine-tunes on XSUM, CNN/DailyMail and WikiHow (100k steps each),
3. generates test-set summaries,
4. copies them into the `evaluation_and_analysis/*_result_files/sbert_pegasus_*_generated_summaries/` folders,
5. runs evaluation steps 1–3 (see Section 8),
6. renames the step 1–3 outputs with the checkpoint (e.g. `..._step3_only_sbert_5M.json`) so the next checkpoint does not overwrite them.

```bash
./run_SBERT_full_pipeline_4M_to_8M.sh          # 5M, 6M, 7M, 8M
./run_SBERT_full_pipeline_4M_to_8M.sh 7 8      # only the listed checkpoints
FORCE=1 ./run_SBERT_full_pipeline_4M_to_8M.sh  # redo every stage
```

Before running it, check the configuration block at the top: the conda environment names (`ENV_TRAIN`, `ENV_STEP1`, `ENV_STEP2`, `ENV_STEP3`), the pretrained model paths, the checkpoint list and the datasets. Set an environment name to `""` to use the currently active environment.

> **This script is meant as an example.** It starts from an existing 4M checkpoint, but it can be extended to start from 1M, with the first checkpoint pretrained from scratch instead of resumed. It can also be adapted for PMI-PEGASUS or ROUGE-PEGASUS. To do that:
> * change `KIND` / `KIND_LOWER` and `PRETRAIN_DATA_DIR` to the other method and its preprocessed dataset;
> * change the evaluation stage so it no longer uses the SBERT-only settings of the step scripts.

## 8. Evaluation Pipeline (Steps 1–5)

All evaluation scripts are in `evaluation_and_analysis/` and are numbered in the order they should be run. **Run them from inside that folder**, since they resolve their paths relative to it.

| Step | Script | Metric | Environment |
| ---- | ------ | ------ | ----------- |
| 1 | `calc_rouge_and_bert_scores_step1.py` | ROUGE-1/2/L and BERTScore (RoBERTa-large) | `pegasus_eval` |
| 2 | `calc_QAeval_metrics_step2.py` | QAEval F1 and is_answered | `pegasus_qaeval` |
| 3 | `calc_BERTscore_deberta_step3.py` | BERTScore with DeBERTa-xlarge-MNLI | `pegasus_eval` |
| 4 | `calc_Llama_scores_like_BERTscore_step4.py` | BERTScore-style similarity using Llama-2-7B embeddings | `pegasus_eval` |
| 5 | `calc_LLM_as_a_judge_step5_Prometheus.py` | Pairwise LLM judge (Prometheus 7B) vs. reference summaries | `prometheus` |
| 5 (SBERT) | `calc_LLM_as_a_judge_step5_Prometheus__for_sbert.py` | Same as step 5, SBERT vs. PMI and SBERT vs. ROUGE | `prometheus` |

### Steps 1–4: Reference-Based Metrics

**Inputs.** For each dataset (`xsum`, `cnn`, `wikihow`), place the files under `evaluation_and_analysis/<dataset>_result_files/`:

```
test_set_<dataset>/dataset.arrow
pmi_pegasus_<dataset>_generated_summaries/generated_predictions.txt
rouge_pegasus_<dataset>_generated_summaries/generated_predictions.txt
sbert_pegasus_<dataset>_generated_summaries/generated_predictions.txt
```

**Chaining.** Each step adds its scores to the previous step's JSON, so they must be run in order:

```
<dataset>_combined_results_for_analysis__step1.json  ->  step2.json  ->  step3.json  ->  step4.json
```

Each JSON is accompanied by a `.log` file holding the printed averages (written by `eval_logging_utils.py`).

**Which models are evaluated** is controlled by flags at the top of each script:

* `eval_for_SBERT` — include SBERT-PEGASUS alongside PMI and ROUGE
* `eval_for_only_sbert` — evaluate only SBERT-PEGASUS; PMI/ROUGE folders are not needed, and output files get an `_only_sbert` suffix so they don't overwrite the combined results

```bash
cd evaluation_and_analysis
python calc_rouge_and_bert_scores_step1.py
python calc_QAeval_metrics_step2.py
python calc_BERTscore_deberta_step3.py
python calc_Llama_scores_like_BERTscore_step4.py
```

Step 4 needs access to `meta-llama/Llama-2-7b-hf`: put your Hugging Face token in `evaluation_and_analysis/HF_TOKEN.txt`.

### Step 5: LLM-as-a-Judge (Prometheus)

These scripts compare two systems' summaries side by side against the dataset's reference summary and ask an LLM which one is better (A / B / tie). They differ from steps 1–4 in how they are used:

* **Inputs are read directly** from the generation outputs, so nothing has to be copied by hand:
  * candidates: `eval_generated_pred/eval_results_{PMI,ROUGE,SBERT}_pegasus_complete_<N>M_pt_100k_ft_<dataset>_comb/generated_predictions.txt`
  * reference summaries: `finetune_data/{xsum_comb,cnn_dailymail_comb,wikihow_comb}`
* **By default they run the full grid** of 3 datasets × 8 checkpoints (1M–8M). Narrow it with `--datasets` and `--checkpoints`.
* **Outputs** go to `<dataset>_result_files/`: one JSON plus one `.log` per comparison, and a per-dataset summary log across all checkpoints.
* **Resumable**: every judged sample is appended to a `.partial.jsonl` file, so an interrupted run continues where it stopped.

* **Step 5** judges PMI vs. ROUGE against the dataset's reference summaries (coverage, faithfulness, conciseness, coherence).
* **Step 5 (SBERT)** does the same, but compares SBERT-PEGASUS against PMI (`sbert_vs_pmi`) and/or ROUGE (`sbert_vs_rouge`).

The model is loaded in 4-bit, so it fits on a 16 GB GPU. Install `bitsandbytes` in the `prometheus` environment in addition to the packages listed in Section 6.

```bash
cd evaluation_and_analysis

python calc_LLM_as_a_judge_step5_Prometheus.py --datasets xsum,cnn --checkpoints 1M,8M
python calc_LLM_as_a_judge_step5_Prometheus__for_sbert.py --pairings sbert_vs_pmi --checkpoints 4M
```

Judging is deterministic (greedy decoding, pinned attention kernel, seeded A/B order), but **only at a fixed `--batch-size`** (default 2). Keep the same batch size for all comparisons you want to put side by side.

## 9. Result Analysis

These scripts in `evaluation_and_analysis/` summarize the evaluation outputs:

* **`analyse_for_detailed_paired_t_tests.py`** — parses the paired t-test result files in `ALL_paired_t_test_results/` (one per checkpoint) and prints per-metric, per-dataset, per-checkpoint significance tables with multiple-comparison correction.
* **`draw_graphs_for_results.py`** — plots PMI-PEGASUS vs. ROUGE-PEGASUS scores across checkpoints for every metric and dataset. The scores are entered in the "static results input area" at the top of the file; update them there before plotting.

## 10. Human Evaluation

The `evaluation_and_analysis/human_eval/` folder contains a pairwise human evaluation of PMI-PEGASUS vs. ROUGE-PEGASUS on 105 examples (35 each from CNN/DailyMail, WikiHow and XSUM, all at the 8M checkpoint). For each example, annotators compared two anonymized summaries and chose which one is **more faithful** and which one is **more informative** (A / B / tie). The annotation set is in `human_eval_set.txt`, and the collected human votes are in `human_answers.csv`.

### GPT Judge Correlation with Human Evaluation

`GPT_API_corralation_check_with_human_eval.py` uses the OpenAI API to have a GPT model judge the same 105 examples. The model sees them in the same A/B order and with the same two questions as the annotators. The script then measures how well the GPT judge agrees with the human annotators.

```bash
pip install openai
export OPENAI_API_KEY="sk-..."

cd evaluation_and_analysis/human_eval
python GPT_API_corralation_check_with_human_eval.py --estimate      # prints the expected cost, sends nothing
python GPT_API_corralation_check_with_human_eval.py --human-answers human_answers.csv
```

The default judge is `gpt-5.6-sol` with reasoning effort `medium`. Change it with `--model` and `--reasoning-effort`. A full run is 210 API calls: two questions for each of the 105 examples. Verdicts that were already collected are cached, so re-running the script does not send them again.

## 11. Maximum Token Lengths (Pretraining, Fine-Tuning, Generation)

Every training and generation script passes two length flags to `src/main.py`:

* `--max_source_length` — input documents are truncated to this many tokens. It is **512** in every script.
* `--max_target_length` — the maximum length of the target text, in tokens. This is the principal sentence during pretraining, the reference summary during fine-tuning, and the generated summary during generation for evaluation.

### Default Values

| Stage | Where it is set | XSUM | CNN/DailyMail | WikiHow |
| ----- | --------------- | ---- | ------------- | ------- |
| Pretraining (C4) | `--max_target_length` in `run_pretrain_pegasus_*.sh` | 256 | 256 | 256 |
| Fine-tuning | `--max_target_length` in `finetune_*_pegasus.sh` | 64 | 128 | 128 |
| Generation for evaluation | `--max_target_length` in `eval_finetuned_models.sh` | 64 | 128 | 128 |

Pretraining does not depend on the downstream dataset, so it uses a single value of 256 for all three. For fine-tuning and generation, use 64 for XSUM (single-sentence summaries) and 128 for CNN/DailyMail and WikiHow (multi-sentence summaries). The PMI and ROUGE results were produced with these values. Keep them identical for all methods (PMI, ROUGE, SBERT) you want to compare, because a lower limit cuts summaries off mid-sentence and lowers every metric.

### Changing the Length per Stage

**Pretraining.** Edit `--max_target_length` in:

* `run_pretrain_pegasus_PMI.sh`
* `run_pretrain_pegasus_ROUGE.sh`
* `run_pretrain_pegasus_SBERT.sh`
* `run_pretrain_pegasus_PMI-from_a_checkpoint.sh`

Principal sentences longer than this are truncated. Use the same value when you resume from a checkpoint as in the original run. In `run_SBERT_full_pipeline_4M_to_8M.sh`, the pretraining value is written directly into the `pretrain_checkpoint()` function.

**Fine-tuning.** Edit `--max_target_length` in `finetune_PMI_pegasus.sh`, `finetune_ROUGE_pegasus.sh` or `finetune_SBERT_pegasus.sh`. Note that the current scripts are meant to be examples. Each example script is set up for one dataset at a time: `finetune_PMI_pegasus.sh` is configured for WikiHow (128), while the ROUGE and SBERT scripts are configured for XSUM (64). When you change `--data_dir` to a different dataset, change `--max_target_length` to match it:

```bash
--data_dir finetune_data/xsum_comb           --max_target_length 64
--data_dir finetune_data/cnn_dailymail_comb  --max_target_length 128
--data_dir finetune_data/wikihow_comb        --max_target_length 128
```

**Generation for evaluation.** Edit `--max_target_length` in `eval_finetuned_models.sh`. It sets the maximum length of the summaries generated for the test set. Use the same value that the model was fine-tuned with: 64 for XSUM, and 128 for CNN/DailyMail and WikiHow.

**Automated SBERT pipeline.** `run_SBERT_full_pipeline_4M_to_8M.sh` sets the length for each dataset once, with `XSUM_TARGET_LEN`, `CNN_TARGET_LEN` and `WIKIHOW_TARGET_LEN` in its configuration block. These values are used for both fine-tuning and generation.

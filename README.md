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





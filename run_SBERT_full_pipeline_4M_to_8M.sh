#!/bin/sh
#
# Full SBERT-Pegasus pipeline, 4M -> 8M, on a fresh machine.
#
# For each of the 5M, 6M, 7M and 8M checkpoints it runs, in order:
#
#   1. pretraining      resumed from the previous checkpoint, 1,000,000 steps
#   2. fine-tuning      100k steps on xsum, cnn_dailymail and wikihow
#   3. generation       do_predict -> eval_generated_pred/.../generated_predictions.txt
#   4. copy             the three prediction files into the sbert_* folders that
#                       evaluation_and_analysis expects to find them in
#   5. steps 1-3        calc_rouge_and_bert_scores_step1.py
#                       calc_QAeval_metrics_step2.py
#                       calc_BERTscore_deberta_step3.py
#   6. archive          rename every file those three wrote so the checkpoint it
#                       belongs to is part of the name, freeing the fixed names
#                       for the next checkpoint
#
# Steps 1-3 write to fixed file names and step2/step3 read the previous step's
# JSON back from those same fixed names, so the renaming in stage 6 can only
# happen once all three have finished.  That is why stage 5 runs all three
# datasets through step1, then all three through step2, then all three through
# step3 -- the same order the scripts themselves use -- instead of finishing one
# dataset at a time.
#
# The three step scripts have eval_for_only_sbert = True, so they read only the
# sbert_* folders and write their output under the "_only_sbert" marker.  No PMI
# or ROUGE prediction files are needed on this machine.
#
# Usage:
#     ./run_SBERT_full_pipeline_4M_to_8M.sh              # 5M, 6M, 7M, 8M
#     ./run_SBERT_full_pipeline_4M_to_8M.sh 7 8          # only those checkpoints
#
# The script is resumable: every stage checks for its own finished output first
# and skips itself when that output is already there, so re-running it after a
# crash picks up where it stopped.  Set FORCE=1 to redo everything regardless.
#
# Prerequisites on this machine:
#   - models/SBERT_pegasus__complete_realnewslike_4_MIL_steps/checkpoint-4000000
#   - PREPROCESSED_DATASETS/c4_realnewslike_processed_SBERT_complete_combined
#   - finetune_data/{xsum_comb,cnn_dailymail_comb,wikihow_comb}
#   - evaluation_and_analysis/*_result_files/test_set_*  (the test set arrow files)
#   - the conda environments listed under "conda environments" below

set -e

###############################################################################
# configuration
###############################################################################

GPU_IDX=0
PORT=29501

KIND=SBERT                 # used in every model / folder name
KIND_LOWER=sbert           # used in the evaluation_and_analysis folder names

# Checkpoints to produce.  Each one resumes from the one before it, so 5 needs
# checkpoint-4000000 to already exist.
DEFAULT_CHECKPOINTS="5 6 7 8"

# --- pretraining ---
PRETRAIN_DATA_DIR=./PREPROCESSED_DATASETS/c4_realnewslike_processed_SBERT_complete_combined
PRETRAIN_SAVE_STEPS=200000
PRETRAIN_WARMUP_STEPS=20000
PRETRAIN_LOGGING_STEP=20000
PRETRAIN_TRAIN_BS=16
PRETRAIN_GRAD_ACC=2
PRETRAIN_LR=1e-4

# Saving every 200,000 steps over 1,000,000 steps leaves 5 checkpoints per run,
# 20 over the whole 4M -> 8M range, each one carrying its DeepSpeed optimizer
# state.  Set this to e.g. 2 to have the Trainer delete the older ones as it
# goes; the checkpoint the next run resumes from is always among the newest, so
# a limit of 2 is safe.  Left empty it keeps all of them, like the existing
# run_pretrain_pegasus_*.sh scripts do.
PRETRAIN_SAVE_TOTAL_LIMIT=""

# --- fine-tuning ---
FT_MAX_STEPS=100000
FT_TAG=100k                # the "_pt_100k_ft_" part of the model folder names
FT_LR=3e-05
FT_LOGGING_STEP=25000
FT_WARMUP_STEPS=500

# --max_target_length per dataset, used for BOTH fine-tuning and generation. It
# must match the PMI / ROUGE runs, or the comparison is not like for like: those
# models were run at 64 for xsum and 128 for cnn and wikihow (their cnn
# predictions reach 125 tokens). A cnn value of 64 truncated 65% of the SBERT
# 4M cnn summaries mid-sentence -- and, being a fine-tuning setting too, the
# training labels with them, so changing it means re-fine-tuning, not just
# regenerating.
XSUM_TARGET_LEN=64
CNN_TARGET_LEN=128
WIKIHOW_TARGET_LEN=128

# --- directories ---
MODELS_DIR=./models
FT_MODELS_DIR=./finetuned_models
PRED_DIR=./eval_generated_pred
ANALYSIS_DIR=./evaluation_and_analysis
LOG_DIR=./pipeline_logs

DATASETS="xsum cnn wikihow"

# --- conda environments ---
# Set any of these to "" to run that stage in whatever environment is already
# active instead of going through `conda run`.
ENV_TRAIN=factP     # pretraining, fine-tuning, generation
ENV_STEP1=factP         # ROUGE + BERTScore
ENV_STEP2=qaeval       # QAEval
ENV_STEP3=llm_score         # DeBERTa BERTScore

###############################################################################
# per-dataset settings
###############################################################################
#
# The target lengths come from the *_TARGET_LEN constants above; the batch size
# / accumulation pairs are the ones the existing scripts use for each dataset.

dataset_config() {
    case "$1" in
        xsum)
            DS_FT_FOLDER=xsum_comb
            DS_EVAL_SUFFIX=xsum_comb
            DS_RESULT_FOLDER=xsum_result_files
            DS_TARGET_LEN=$XSUM_TARGET_LEN
            DS_TRAIN_BS=16
            DS_GRAD_ACC=2
            ;;
        cnn)
            DS_FT_FOLDER=cnn_dailymail_comb
            DS_EVAL_SUFFIX=cnn_comb
            DS_RESULT_FOLDER=cnn_result_files
            DS_TARGET_LEN=$CNN_TARGET_LEN
            DS_TRAIN_BS=16
            DS_GRAD_ACC=2
            ;;
        wikihow)
            DS_FT_FOLDER=wikihow_comb
            DS_EVAL_SUFFIX=wikihow_comb
            DS_RESULT_FOLDER=wikihow_result_files
            DS_TARGET_LEN=$WIKIHOW_TARGET_LEN
            DS_TRAIN_BS=8
            DS_GRAD_ACC=4
            ;;
        *)
            echo "unknown dataset: $1" >&2
            exit 1
            ;;
    esac
    DS_SUMMARY_FOLDER="${KIND_LOWER}_pegasus_${1}_generated_summaries"
}

###############################################################################
# helpers
###############################################################################

say() {
    echo ""
    echo "==============================================================="
    echo "  $*"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==============================================================="
}

# Runs a command in a conda environment, mirrors its output into a log file and
# fails the script when the command fails.  `tee` would otherwise swallow the
# exit status, so it is written to a temporary file and read back.
run_stage() {
    _log="$1"; shift
    _env="$1"; shift

    mkdir -p "$(dirname "$_log")"
    _status_file="${_log}.status"
    rm -f "$_status_file"

    {
        if [ -z "$_env" ]; then
            "$@"
        else
            conda run --no-capture-output -n "$_env" "$@"
        fi
        echo $? > "$_status_file"
    } 2>&1 | tee -a "$_log"

    _status=$(cat "$_status_file" 2>/dev/null || echo 1)
    rm -f "$_status_file"

    if [ "$_status" -ne 0 ]; then
        echo "" >&2
        echo "FAILED (exit $_status): $*" >&2
        echo "see $_log" >&2
        exit "$_status"
    fi
}

# A saved model directory, regardless of which serialization format was used.
model_is_saved() {
    [ -f "$1/pytorch_model.bin" ] || [ -f "$1/model.safetensors" ]
}

###############################################################################
# stage 1 -- pretraining
###############################################################################

pretrain_checkpoint() {
    n="$1"
    prev=$((n - 1))

    prev_ckpt="$MODELS_DIR/${KIND}_pegasus__complete_realnewslike_${prev}_MIL_steps/checkpoint-${prev}000000"
    out_dir="$MODELS_DIR/${KIND}_pegasus__complete_realnewslike_${n}_MIL_steps"

    if [ "$FORCE" != "1" ] && model_is_saved "$out_dir"; then
        say "pretrain ${n}M -- already done, skipping"
        return 0
    fi

    if [ ! -d "$prev_ckpt" ]; then
        echo "missing checkpoint to resume from: $prev_ckpt" >&2
        exit 1
    fi

    say "pretrain ${n}M  (resume from checkpoint-${prev}000000, up to ${n},000,000 steps)"

    _limit=""
    if [ -n "$PRETRAIN_SAVE_TOTAL_LIMIT" ]; then
        _limit="--save_total_limit $PRETRAIN_SAVE_TOTAL_LIMIT"
    fi

    run_stage "$LOG_DIR/${n}M/pretrain.log" "$ENV_TRAIN" \
        deepspeed --include=localhost:"$GPU_IDX" src/main.py --fp16 \
        --data_dir "$PRETRAIN_DATA_DIR" \
        --do_train --do_pretrain --model_name facebook/bart-base \
        --resume_from_checkpoint "$prev_ckpt" \
        --deepspeed src/ds_config.json \
        --per_device_train_batch_size $PRETRAIN_TRAIN_BS \
        --gradient_accumulation_steps $PRETRAIN_GRAD_ACC \
        --learning_rate $PRETRAIN_LR --weight_decay 0.01 \
        --logging_step $PRETRAIN_LOGGING_STEP --max_steps ${n}000000 \
        --warmup_steps $PRETRAIN_WARMUP_STEPS --save_steps $PRETRAIN_SAVE_STEPS \
        $_limit \
        --max_source_length 512 --max_target_length 256 \
        --output_dir "$out_dir" --pretrain_model_type bart_base --tokenize_on_fly
}

###############################################################################
# stage 2 -- fine-tuning
###############################################################################

finetune_dataset() {
    n="$1"
    ds="$2"
    dataset_config "$ds"

    pretrained="$MODELS_DIR/${KIND}_pegasus__complete_realnewslike_${n}_MIL_steps"
    out_dir="$FT_MODELS_DIR/${KIND}_pegasus_complete_${n}M_pt_${FT_TAG}_ft_${DS_EVAL_SUFFIX}"

    if [ "$FORCE" != "1" ] && model_is_saved "$out_dir"; then
        say "finetune ${n}M / ${ds} -- already done, skipping"
        return 0
    fi

    say "finetune ${n}M / ${ds}  (${FT_MAX_STEPS} steps, target length ${DS_TARGET_LEN})"

    run_stage "$LOG_DIR/${n}M/finetune_${ds}.log" "$ENV_TRAIN" \
        deepspeed --master_port=$PORT --include=localhost:$GPU_IDX src/main.py --fp16 \
        --deepspeed src/ds_config.json \
        --data_dir "finetune_data/$DS_FT_FOLDER" --do_finetune \
        --do_train --model_name "$pretrained" \
        --evaluation_strategy no \
        --per_device_train_batch_size $DS_TRAIN_BS --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps $DS_GRAD_ACC \
        --learning_rate $FT_LR --weight_decay 0.01 --label_smoothing 0.1 \
        --max_source_length 512 --max_target_length $DS_TARGET_LEN \
        --logging_step $FT_LOGGING_STEP --max_steps $FT_MAX_STEPS \
        --warmup_steps $FT_WARMUP_STEPS --save_steps $FT_MAX_STEPS \
        --output_dir "$out_dir" \
        --contrastive_learning --pertubation_type intrinsic --num_negatives 5 --contrastive_weight 5
}

###############################################################################
# stage 3 -- generation
###############################################################################

generate_predictions() {
    n="$1"
    ds="$2"
    dataset_config "$ds"

    ft_model="$FT_MODELS_DIR/${KIND}_pegasus_complete_${n}M_pt_${FT_TAG}_ft_${DS_EVAL_SUFFIX}"
    out_dir="$PRED_DIR/eval_results_${KIND}_pegasus_complete_${n}M_pt_${FT_TAG}_ft_${DS_EVAL_SUFFIX}"

    if [ "$FORCE" != "1" ] && [ -s "$out_dir/generated_predictions.txt" ]; then
        say "generate ${n}M / ${ds} -- already done, skipping"
        return 0
    fi

    say "generate ${n}M / ${ds}"

    run_stage "$LOG_DIR/${n}M/generate_${ds}.log" "$ENV_TRAIN" \
        python src/main.py --fp16 \
        --data_dir "finetune_data/$DS_FT_FOLDER" --do_predict --predict_with_generate \
        --model_name "$ft_model" \
        --per_device_train_batch_size 16 --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --learning_rate $FT_LR --weight_decay 0.01 --label_smoothing 0.1 \
        --max_source_length 512 --max_target_length $DS_TARGET_LEN \
        --logging_step 1000 \
        --output_dir "$out_dir"

    if [ ! -s "$out_dir/generated_predictions.txt" ]; then
        echo "generation produced no $out_dir/generated_predictions.txt" >&2
        exit 1
    fi
}

###############################################################################
# stage 4 -- put the predictions where steps 1-3 look for them
###############################################################################
#
# The step scripts read a hard-coded
#   <ds>_result_files/sbert_pegasus_<ds>_generated_summaries/generated_predictions.txt
# so each checkpoint's file is copied over that fixed name in turn.  A second
# copy carrying the checkpoint in its name is kept next to it, so the summaries
# a given set of scores came from stay identifiable afterwards.

stage_predictions() {
    n="$1"
    ds="$2"
    dataset_config "$ds"

    src="$PRED_DIR/eval_results_${KIND}_pegasus_complete_${n}M_pt_${FT_TAG}_ft_${DS_EVAL_SUFFIX}/generated_predictions.txt"
    dest_dir="$ANALYSIS_DIR/$DS_RESULT_FOLDER/$DS_SUMMARY_FOLDER"

    if [ ! -s "$src" ]; then
        echo "missing predictions to copy: $src" >&2
        exit 1
    fi

    mkdir -p "$dest_dir"
    cp "$src" "$dest_dir/generated_predictions.txt"
    cp "$src" "$dest_dir/generated_predictions_${n}M.txt"

    echo "  ${ds}: $(wc -l < "$dest_dir/generated_predictions.txt") summaries -> $dest_dir/generated_predictions.txt"
}

###############################################################################
# stage 5 -- the three analysis scripts
###############################################################################
#
# Each one walks xsum, cnn and wikihow itself, and each one reads the JSON the
# previous one wrote, so they run in order and none of them may be renamed until
# step3 has finished.

# The three scripts resolve every path they use relative to their own folder, so
# they are run from inside it.  LOG_DIR was made absolute up front precisely so
# that the log still lands in the right place from in there.
run_analysis_steps() {
    n="$1"

    say "step1 (ROUGE + BERTScore) for ${n}M"
    ( cd "$ANALYSIS_DIR" && run_stage "$LOG_DIR/${n}M/step1.log" "$ENV_STEP1" \
        python calc_rouge_and_bert_scores_step1.py )

    say "step2 (QAEval) for ${n}M"
    ( cd "$ANALYSIS_DIR" && run_stage "$LOG_DIR/${n}M/step2.log" "$ENV_STEP2" \
        python calc_QAeval_metrics_step2.py )

    say "step3 (DeBERTa BERTScore) for ${n}M"
    ( cd "$ANALYSIS_DIR" && run_stage "$LOG_DIR/${n}M/step3.log" "$ENV_STEP3" \
        python calc_BERTscore_deberta_step3.py )
}

###############################################################################
# stage 6 -- rename the results so the checkpoint is part of the name
###############################################################################
#
#   xsum_combined_results_for_analysis__step3_only_sbert.json
#       -> xsum_combined_results_for_analysis__step3_only_sbert_5M.json
#
# Both the .json and the .log of all three steps are renamed, for all three
# datasets: 18 files per checkpoint.  Without this the next checkpoint would
# overwrite every one of them.

archive_analysis_outputs() {
    n="$1"
    say "archiving step1-3 results as ${n}M"

    for ds in $DATASETS; do
        dataset_config "$ds"
        for s in 1 2 3; do
            for ext in json log; do
                base="$ANALYSIS_DIR/$DS_RESULT_FOLDER/${ds}_combined_results_for_analysis__step${s}_only_sbert"
                if [ ! -f "$base.$ext" ]; then
                    echo "expected output missing: $base.$ext" >&2
                    exit 1
                fi
                mv "$base.$ext" "${base}_${n}M.$ext"
                echo "  ${ds}_combined_results_for_analysis__step${s}_only_sbert_${n}M.$ext"
            done
        done
    done
}

# True once every step3 result for this checkpoint has been archived, which is
# what makes the analysis stage skippable on a re-run.
analysis_already_done() {
    n="$1"
    for ds in $DATASETS; do
        dataset_config "$ds"
        f="$ANALYSIS_DIR/$DS_RESULT_FOLDER/${ds}_combined_results_for_analysis__step3_only_sbert_${n}M.json"
        [ -f "$f" ] || return 1
    done
    return 0
}

###############################################################################
# main
###############################################################################

if [ $# -gt 0 ]; then
    CHECKPOINTS="$*"
else
    CHECKPOINTS="$DEFAULT_CHECKPOINTS"
fi

mkdir -p "$LOG_DIR"
# Absolute, so that the stages that have to run from inside evaluation_and_analysis
# still write their logs here.
LOG_DIR=$(cd "$LOG_DIR" && pwd)

say "SBERT pipeline -- checkpoints: $CHECKPOINTS"
echo "  pretraining data : $PRETRAIN_DATA_DIR"
echo "  save_steps       : $PRETRAIN_SAVE_STEPS"
echo "  datasets         : $DATASETS"
echo "  logs             : $LOG_DIR"
if [ "$FORCE" = "1" ]; then
    echo "  FORCE=1          : finished stages will be redone"
fi

for n in $CHECKPOINTS; do
    mkdir -p "$LOG_DIR/${n}M"

    pretrain_checkpoint "$n"

    for ds in $DATASETS; do
        finetune_dataset "$n" "$ds"
        generate_predictions "$n" "$ds"
    done

    if [ "$FORCE" != "1" ] && analysis_already_done "$n"; then
        say "step1-3 for ${n}M -- already archived, skipping"
    else
        say "staging ${n}M predictions for evaluation_and_analysis"
        for ds in $DATASETS; do
            stage_predictions "$n" "$ds"
        done

        run_analysis_steps "$n"
        archive_analysis_outputs "$n"
    fi

    say "checkpoint ${n}M complete"
done

say "all done -- checkpoints: $CHECKPOINTS"

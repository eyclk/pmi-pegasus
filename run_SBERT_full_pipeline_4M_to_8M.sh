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
#     ./run_SBERT_full_pipeline_4M_to_8M.sh --analysis-only
#     ./run_SBERT_full_pipeline_4M_to_8M.sh --analysis-only --datasets wikihow 5 6 7 8
#     ./run_SBERT_full_pipeline_4M_to_8M.sh --eval-only --datasets wikihow
#
# Three levels of re-running, from most to least work:
#
#   (no flag)         pretrain -> fine-tune -> generate -> steps 1-3
#   --eval-only                               generate -> steps 1-3
#   --analysis-only                                       steps 1-3
#
# --eval-only (equivalently EVAL_ONLY=1) skips pretraining and fine-tuning and
# regenerates the summaries in eval_generated_pred from the fine-tuned models
# that are already in finetuned_models, then runs steps 1-3 on them. Both the
# generated_predictions.txt files and the step 1-3 results of the same
# checkpoint and dataset are replaced in place. Use it after changing something
# about generation (e.g. --max_target_length, or src/main.py's decoding). It
# always regenerates, even when a generated_predictions.txt is already there --
# that is the difference from a plain re-run, which would skip generation.
#
# Because the summaries change, any step 5 judge results of that checkpoint
# and dataset no longer describe them -- and step 5 skips a comparison whose
# output already exists, and resumes a .partial.jsonl without checking what it
# judged, so it would quietly keep the old verdicts. --eval-only therefore moves
# those files aside (renamed with "__before_regen_<timestamp>", never deleted)
# so the next step 5 run judges the new summaries.
#
# --datasets (equivalently DATASETS="...") limits every per-dataset stage --
# fine-tuning, generation, copying, steps 1-3 and renaming -- to the datasets
# given, comma- or space-separated (e.g. "wikihow" or "cnn,wikihow"); the
# default is all three. The results of the other datasets are left untouched.
# Pretraining is per checkpoint, not per dataset, so it is unaffected.
#
# --analysis-only (equivalently ANALYSIS_ONLY=1) re-runs stages 4-6 on the
# summaries that are already in eval_generated_pred and overwrites the step 1-3
# results of the same checkpoint and dataset in place.  Pretraining, fine-tuning
# and generation are skipped entirely: no model is trained, no summary is
# regenerated, and the summaries the new scores are computed from are the very
# ones the old scores were computed from.  Use it after changing something
# inside the step 1-3 scripts (a metric, a batch size, a model) rather than the
# summaries.
#
# The script is resumable: every stage checks for its own finished output first
# and skips itself when that output is already there, so re-running it after a
# crash picks up where it stopped.  Set FORCE=1 to redo everything regardless.
# An interrupted pretraining stage resumes from the newest complete
# checkpoint-* it saved (every PRETRAIN_SAVE_STEPS), not from the start of the
# stage -- e.g. a 5M stage that died after checkpoint-4400000 continues from
# 4,400,000 rather than 4,000,000.
# --analysis-only deliberately does NOT skip a checkpoint whose step 1-3 results
# are already there, because replacing exactly those is the point of it.
#
# Prerequisites on this machine:
#   - models/SBERT_pegasus__complete_realnewslike_4_MIL_steps/checkpoint-4000000
#   - PREPROCESSED_DATASETS/c4_realnewslike_processed_SBERT_complete_combined
#   - finetune_data/{xsum_comb,cnn_dailymail_comb,wikihow_comb}
#   - evaluation_and_analysis/*_result_files/test_set_*  (the test set arrow files)
#   - the conda environments listed under "conda environments" below
#   With --analysis-only only the test sets and the step 1-3 environments are
#   needed; the pretraining set and the fine-tuning data are not touched.

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

# Skip pretraining, fine-tuning and generation and only re-run steps 1-3,
# overwriting the step 1-3 results of the same checkpoint and dataset.  Set it
# here, pass --analysis-only, or put ANALYSIS_ONLY=1 in front of the command.
ANALYSIS_ONLY=${ANALYSIS_ONLY:-0}

# Skip pretraining and fine-tuning, regenerate the summaries from the existing
# fine-tuned models, then re-run steps 1-3, replacing both in place. Set it
# here, pass --eval-only, or put EVAL_ONLY=1 in front of the command.
EVAL_ONLY=${EVAL_ONLY:-0}

# Appended, under --eval-only, to the step 5 results of the summaries being
# replaced. Timestamped, so repeated --eval-only runs never collide.
STALE_SUFFIX="__before_regen_$(date '+%Y%m%d-%H%M%S')"

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

# Which datasets to run. Overridden by --datasets, or by DATASETS="..." in front
# of the command.
DATASETS=${DATASETS:-"xsum cnn wikihow"}

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
        # This group runs in a subshell that inherits `set -e`, which would end
        # it on a failure before the status below is recorded -- reporting every
        # failure as exit 1 instead of the command's own code.
        set +e
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

# Prints the newest checkpoint-* folder in $1 that finished saving, or nothing.
#
# A crash during a save leaves a half-written checkpoint-* folder behind, which
# the Trainer would happily try to resume from and fail on. The Trainer writes
# trainer_state.json after the model weights and the DeepSpeed state, so a
# folder that has it is complete. Incomplete ones are skipped, newest first,
# until a complete one is found.
latest_complete_checkpoint() {
    for _step in $(ls -d "$1"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -rn); do
        if [ -f "$1/checkpoint-$_step/trainer_state.json" ]; then
            echo "$1/checkpoint-$_step"
            return 0
        fi
        echo "  skipping incomplete checkpoint (no trainer_state.json): $1/checkpoint-$_step" >&2
    done
}

# Under --eval-only, renames the step 5 judge results of checkpoint $1 and
# dataset $2 with STALE_SUFFIX before their summaries are regenerated:
#   <ds>_<n>M_sbert_vs_{pmi,rouge}_llm_judge_vs_reference_summaries__step5.{json,log,partial.jsonl}
#     -> ..._step5__before_regen_<timestamp>.{json,log,partial.jsonl}
# The step 5 summary logs are left alone: step 5 rebuilds them from the
# per-checkpoint files on disk on its next run.
move_aside_stale_step5() {
    n="$1"
    ds="$2"
    dataset_config "$ds"

    _moved=""
    for opponent in pmi rouge; do
        base="$ANALYSIS_DIR/$DS_RESULT_FOLDER/${ds}_${n}M_${KIND_LOWER}_vs_${opponent}_llm_judge_vs_reference_summaries__step5"
        for ext in json log partial.jsonl; do
            [ -e "$base.$ext" ] || continue
            if [ -z "$_moved" ]; then
                say "step 5 results of ${n}M / ${ds} judged the old summaries -- moving them aside"
                _moved=1
            fi
            dst="$base$STALE_SUFFIX.$ext"
            if [ -e "$dst" ]; then
                echo "refusing to overwrite $dst while moving $base.$ext aside" >&2
                exit 1
            fi
            mv "$base.$ext" "$dst"
            echo "  $base.$ext"
            echo "    -> $dst"
        done
    done
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

    # A run interrupted part way through this stage left its own checkpoints in
    # out_dir (every PRETRAIN_SAVE_STEPS), and the newest complete one is where
    # it should pick up. src/main.py would find it by itself, but only when
    # --resume_from_checkpoint is not given -- an explicit one always wins -- so
    # the choice has to be made here. With none in out_dir the stage is starting
    # fresh, from the end of the previous one. FORCE=1 also starts fresh.
    resume_from=""
    if [ "$FORCE" != "1" ]; then
        resume_from=$(latest_complete_checkpoint "$out_dir")
    fi

    if [ -n "$resume_from" ]; then
        say "pretrain ${n}M  (continuing interrupted run from ${resume_from##*/}, up to ${n},000,000 steps)"
    else
        if [ ! -d "$prev_ckpt" ]; then
            echo "missing checkpoint to resume from: $prev_ckpt" >&2
            exit 1
        fi
        resume_from="$prev_ckpt"
        say "pretrain ${n}M  (resume from checkpoint-${prev}000000, up to ${n},000,000 steps)"
    fi

    _limit=""
    if [ -n "$PRETRAIN_SAVE_TOTAL_LIMIT" ]; then
        _limit="--save_total_limit $PRETRAIN_SAVE_TOTAL_LIMIT"
    fi

    run_stage "$LOG_DIR/${n}M/pretrain.log" "$ENV_TRAIN" \
        deepspeed --include=localhost:"$GPU_IDX" src/main.py --fp16 \
        --data_dir "$PRETRAIN_DATA_DIR" \
        --do_train --do_pretrain --model_name facebook/bart-base \
        --resume_from_checkpoint "$resume_from" \
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

    # --eval-only exists to regenerate, so it never takes this skip.
    if [ "$FORCE" != "1" ] && [ "$EVAL_ONLY" != "1" ] && [ -s "$out_dir/generated_predictions.txt" ]; then
        say "generate ${n}M / ${ds} -- already done, skipping"
        return 0
    fi

    say "generate ${n}M / ${ds}"

    # src/main.py only writes this file once every summary is generated, so
    # without removing the old one first, the check after the run below could
    # not tell new output from the previous run's.
    rm -f "$out_dir/generated_predictions.txt"

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
# Each one reads the JSON the previous one wrote, so they run in order and none
# of them may be renamed until step3 has finished.
#
# Their __main__ always runs xsum, cnn and wikihow, which with a --datasets
# subset would score the other datasets on whatever stale predictions sit in
# their sbert_* folders. So they are imported instead and only the functions for
# DATASETS are called. Importing still runs their module-level setup (flags,
# model preloading), which is what their functions rely on. sys.argv[0] is set
# to the script's name first, so the header eval_logging_utils writes into each
# .log names the script, as it did when the scripts were run directly.

# $1 = module, $2 = function prefix
#   -> "import sys; sys.argv[0] = 'm.py'; import m; m.<prefix>wikihow()"
step_call() {
    _code="import sys; sys.argv[0] = '$1.py'; import $1"
    for ds in $DATASETS; do
        _code="$_code; $1.$2$ds()"
    done
    echo "$_code"
}

# The three scripts resolve every path they use relative to their own folder, so
# they are run from inside it.  LOG_DIR was made absolute up front precisely so
# that the log still lands in the right place from in there.
run_analysis_steps() {
    n="$1"

    say "step1 (ROUGE + BERTScore) for ${n}M -- $DATASETS"
    ( cd "$ANALYSIS_DIR" && run_stage "$LOG_DIR/${n}M/step1.log" "$ENV_STEP1" \
        python -c "$(step_call calc_rouge_and_bert_scores_step1 combine_results_of_)" )

    say "step2 (QAEval) for ${n}M -- $DATASETS"
    ( cd "$ANALYSIS_DIR" && run_stage "$LOG_DIR/${n}M/step2.log" "$ENV_STEP2" \
        python -c "$(step_call calc_QAeval_metrics_step2 calc_qaeval_metric_of_)" )

    say "step3 (DeBERTa BERTScore) for ${n}M -- $DATASETS"
    ( cd "$ANALYSIS_DIR" && run_stage "$LOG_DIR/${n}M/step3.log" "$ENV_STEP3" \
        python -c "$(step_call calc_BERTscore_deberta_step3 calc_deberta_f1_metric_of_)" )
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
                # Under --analysis-only, --eval-only or FORCE=1 the destination already
                # holds the previous run's numbers and `mv` replaces them, which
                # is the point of those modes, so it is reported rather than
                # refused.
                _note=""
                [ -f "${base}_${n}M.$ext" ] && _note="   (replaced)"
                mv "$base.$ext" "${base}_${n}M.$ext"
                echo "  ${ds}_combined_results_for_analysis__step${s}_only_sbert_${n}M.$ext$_note"
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

CHECKPOINTS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --analysis-only) ANALYSIS_ONLY=1 ;;
        --eval-only) EVAL_ONLY=1 ;;
        --datasets)
            if [ $# -lt 2 ]; then
                echo "--datasets needs a value, e.g. --datasets wikihow" >&2
                exit 1
            fi
            DATASETS="$2"
            shift
            ;;
        --datasets=*) DATASETS="${1#--datasets=}" ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        -*)
            echo "unknown option: $1" >&2
            echo "usage: $0 [--eval-only | --analysis-only] [--datasets xsum,cnn,wikihow] [checkpoint ...]" >&2
            exit 1
            ;;
        *) CHECKPOINTS="$CHECKPOINTS $1" ;;
    esac
    shift
done

if [ -z "$CHECKPOINTS" ]; then
    CHECKPOINTS="$DEFAULT_CHECKPOINTS"
fi

# One says regenerate the summaries, the other says keep them; there is no
# sensible run that does both.
if [ "$EVAL_ONLY" = "1" ] && [ "$ANALYSIS_ONLY" = "1" ]; then
    echo "--eval-only and --analysis-only cannot be combined:" >&2
    echo "  --eval-only     regenerates the summaries, then runs steps 1-3" >&2
    echo "  --analysis-only keeps the summaries, and only runs steps 1-3" >&2
    exit 1
fi

# "cnn,wikihow" and "cnn wikihow" both become "cnn wikihow", with repeats
# dropped -- a dataset listed twice would be scored twice and its second rename
# would find nothing to rename. Every name is checked here, before anything
# runs, rather than when its first stage comes up.
DATASETS=$(echo "$DATASETS" | tr ', ' '\n\n' | awk 'NF && !seen[$0]++' | xargs)
if [ -z "$DATASETS" ]; then
    echo "--datasets is empty" >&2
    exit 1
fi
for ds in $DATASETS; do
    case "$ds" in
        xsum|cnn|wikihow) ;;
        *) echo "unknown dataset: $ds (choose from xsum, cnn, wikihow)" >&2; exit 1 ;;
    esac
done

# --analysis-only scores summaries that must already be there, so a missing one
# is caught up front rather than after the earlier checkpoints have been scored.
if [ "$ANALYSIS_ONLY" = "1" ]; then
    for n in $CHECKPOINTS; do
        for ds in $DATASETS; do
            dataset_config "$ds"
            preds="$PRED_DIR/eval_results_${KIND}_pegasus_complete_${n}M_pt_${FT_TAG}_ft_${DS_EVAL_SUFFIX}/generated_predictions.txt"
            if [ ! -s "$preds" ]; then
                echo "--analysis-only needs summaries that are already there, but" >&2
                echo "$preds is missing or empty." >&2
                echo "Run without --analysis-only to train and generate them first." >&2
                exit 1
            fi
            echo "  ${n}M / ${ds}: $preds"
        done
    done
fi

# --eval-only generates from fine-tuned models that must already be there, for
# the same reason.
if [ "$EVAL_ONLY" = "1" ]; then
    for n in $CHECKPOINTS; do
        for ds in $DATASETS; do
            dataset_config "$ds"
            ft_model="$FT_MODELS_DIR/${KIND}_pegasus_complete_${n}M_pt_${FT_TAG}_ft_${DS_EVAL_SUFFIX}"
            if ! model_is_saved "$ft_model"; then
                echo "--eval-only needs fine-tuned models that are already there, but" >&2
                echo "$ft_model holds no pytorch_model.bin / model.safetensors." >&2
                echo "Run without --eval-only to fine-tune it first." >&2
                exit 1
            fi
            echo "  ${n}M / ${ds}: $ft_model"
        done
    done
fi

mkdir -p "$LOG_DIR"
# Absolute, so that the stages that have to run from inside evaluation_and_analysis
# still write their logs here.
LOG_DIR=$(cd "$LOG_DIR" && pwd)

if [ "$ANALYSIS_ONLY" = "1" ]; then
    say "SBERT steps 1-3 only -- checkpoints: $CHECKPOINTS"
    echo "  datasets         : $DATASETS"
    echo "  pretraining      : skipped (--analysis-only)"
    echo "  fine-tune        : skipped (--analysis-only)"
    echo "  generation       : skipped (--analysis-only)"
    echo "  step 1-3 results : replaced in place"
    echo "  logs             : $LOG_DIR"
elif [ "$EVAL_ONLY" = "1" ]; then
    say "SBERT generation + steps 1-3 -- checkpoints: $CHECKPOINTS"
    echo "  datasets         : $DATASETS"
    echo "  pretraining      : skipped (--eval-only)"
    echo "  fine-tune        : skipped (--eval-only)"
    echo "  generation       : rerun from finetuned_models, summaries replaced in place"
    echo "  step 1-3 results : replaced in place"
    echo "  step 5 results   : moved aside with \"$STALE_SUFFIX\""
    echo "  logs             : $LOG_DIR"
else
    say "SBERT pipeline -- checkpoints: $CHECKPOINTS"
    echo "  pretraining data : $PRETRAIN_DATA_DIR"
    echo "  save_steps       : $PRETRAIN_SAVE_STEPS"
    echo "  datasets         : $DATASETS"
    echo "  logs             : $LOG_DIR"
    if [ "$FORCE" = "1" ]; then
        echo "  FORCE=1          : finished stages will be redone"
    fi
fi

for n in $CHECKPOINTS; do
    mkdir -p "$LOG_DIR/${n}M"

    if [ "$EVAL_ONLY" = "1" ]; then
        for ds in $DATASETS; do
            move_aside_stale_step5 "$n" "$ds"
            generate_predictions "$n" "$ds"
        done
    elif [ "$ANALYSIS_ONLY" != "1" ]; then
        pretrain_checkpoint "$n"

        for ds in $DATASETS; do
            finetune_dataset "$n" "$ds"
            generate_predictions "$n" "$ds"
        done
    fi

    # Under --analysis-only and --eval-only the results that are already there
    # are precisely the ones being replaced, so the skip does not apply.
    if [ "$ANALYSIS_ONLY" != "1" ] && [ "$EVAL_ONLY" != "1" ] && [ "$FORCE" != "1" ] \
            && analysis_already_done "$n"; then
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

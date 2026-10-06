#!/bin/sh
#
# Re-fine-tunes and re-evaluates the SBERT-Pegasus 1M -> 4M checkpoints on cnn
# and wikihow, with a target length of 128 for both.
#
# Why: run_SBERT_full_pipeline_4M_to_8M.sh used to fine-tune AND generate cnn at
# --max_target_length 64, while the PMI / ROUGE models were run at 128. 65% of
# the SBERT 4M cnn summaries ended mid-sentence, which made every SBERT cnn
# metric (and the step 5 judge) incomparable with PMI / ROUGE. Because 64 was a
# fine-tuning setting too -- the training labels were truncated as well --
# regenerating is not enough; the models have to be fine-tuned again. xsum was
# run at 64 like PMI / ROUGE and is left alone.
#
# For each checkpoint and each of cnn / wikihow it runs, in order:
#
#   0. archive          renames what an earlier run left behind by appending
#                       "__version1" to the name (see archive_version1 below),
#                       so nothing is overwritten and the old results stay
#                       available next to the new ones
#   1. fine-tuning      100k steps, target length CNN_TARGET_LEN / WIKIHOW_TARGET_LEN
#   2. generation       do_predict -> eval_generated_pred/.../generated_predictions.txt
#   3. copy             into the sbert_* folders evaluation_and_analysis reads
#   4. steps 1-3        ROUGE + BERTScore, QAEval, DeBERTa BERTScore -- cnn and
#                       wikihow only (the step scripts are imported and only
#                       their cnn / wikihow functions are called; the scripts
#                       themselves are not modified)
#   5. archive          step 1-3 outputs renamed with the checkpoint, exactly as
#                       run_SBERT_full_pipeline_4M_to_8M.sh names them
#
# Step 5 (the LLM judge) is NOT part of this script; run it afterwards with
#   python calc_LLM_as_a_judge_step5_Prometheus__for_sbert.py \
#       --datasets cnn,wikihow --checkpoints 1M,2M,3M,4M
#
# Usage:
#     ./run_SBERT_refinetune_cnn_wikihow_1M_to_4M.sh            # 1M, 2M, 3M, 4M
#     ./run_SBERT_refinetune_cnn_wikihow_1M_to_4M.sh 3 4        # only those checkpoints
#     ./run_SBERT_refinetune_cnn_wikihow_1M_to_4M.sh --analysis-only
#     ./run_SBERT_refinetune_cnn_wikihow_1M_to_4M.sh --eval-only
#
# Three levels of re-running, from most to least work:
#
#   (no flag)         archive -> fine-tune -> generate -> steps 1-3
#   --eval-only                               generate -> steps 1-3
#   --analysis-only                                       steps 1-3
#
# --eval-only (equivalently EVAL_ONLY=1) regenerates the summaries in
# eval_generated_pred from the fine-tuned models already in finetuned_models,
# then runs steps 1-3 on them. Both the generated_predictions.txt files and the
# step 1-3 results of the same checkpoint and dataset are replaced in place.
# Stages 0 and 1 are skipped: stage 0 would move away the very fine-tuned model
# generation needs. Use it after changing something about generation (e.g.
# --max_target_length, or src/main.py's decoding). It always regenerates, even
# when a generated_predictions.txt is already there.
#
# Because the summaries change, any step 5 judge results of that checkpoint
# and dataset no longer describe them -- and step 5 skips a comparison whose
# output already exists, and resumes a .partial.jsonl without checking what it
# judged, so it would quietly keep the old verdicts. --eval-only therefore moves
# those files aside (renamed with "__before_regen_<timestamp>", never deleted)
# so the next step 5 run judges the new summaries.
#
# --analysis-only (equivalently ANALYSIS_ONLY=1) re-runs steps 1-3 on the
# summaries that are already in eval_generated_pred and overwrites the step 1-3
# results of the same checkpoint and dataset in place. Stages 0, 1 and 2 are
# skipped entirely: nothing is fine-tuned, nothing is generated, and nothing is
# renamed to "__version1" -- the summaries the new scores are computed from are
# the very ones the old scores were computed from, so the old scores are not a
# separate version worth keeping. Use it after changing something inside the
# step 1-3 scripts (a metric, a batch size, a model) rather than the summaries.
#
# The script is resumable: the archive step runs once per checkpoint and dataset
# (a marker file in LOG_DIR records it), and every later stage skips itself when
# its own output is already there. Re-running after a crash therefore picks up
# where it stopped and never renames the NEW results as "__version1".
# --analysis-only and --eval-only deliberately do NOT skip a checkpoint whose
# step 1-3 results are already there, because replacing exactly those is the
# point of them.
#
# Prerequisites on this machine:
#   - the SBERT pretrained models for the requested checkpoints, at the
#     PRETRAINED_<n>M paths set under "pretrained models" below
#     (not needed with --analysis-only or --eval-only, which never fine-tune)
#   - finetune_data/{cnn_dailymail_comb,wikihow_comb}
#   - the conda environments listed under "conda environments" below

set -e

###############################################################################
# configuration
###############################################################################

GPU_IDX=0
PORT=29501

KIND=SBERT                 # used in every model / folder name
KIND_LOWER=sbert           # used in the evaluation_and_analysis folder names

DEFAULT_CHECKPOINTS="1 2 3"
DATASETS="wikihow"

# Skip fine-tuning and generation and only re-run steps 1-3, overwriting the
# step 1-3 results of the same checkpoint and dataset. Set it here, pass
# --analysis-only, or put ANALYSIS_ONLY=1 in front of the command.
ANALYSIS_ONLY=${ANALYSIS_ONLY:-0}

# Skip archiving and fine-tuning, regenerate the summaries from the existing
# fine-tuned models, then re-run steps 1-3, replacing both in place. Set it
# here, pass --eval-only, or put EVAL_ONLY=1 in front of the command.
EVAL_ONLY=${EVAL_ONLY:-0}

# --- pretrained models ---
# The model each checkpoint is fine-tuned from. Either a final model folder or a
# checkpoint-* folder inside a pretraining run, e.g.
#   ./models/SBERT_pegasus__complete_realnewslike_3_MIL_steps/checkpoint-1000000
# Each one must hold the model AND its tokenizer (src/main.py loads both from
# it), and its trainer_state.json must say it is at n million steps. All the
# requested paths are checked before anything is renamed or trained.
PRETRAINED_1M=./models/SBERT_pegasus__complete_realnewslike_1_MIL_steps/checkpoint-1000000
PRETRAINED_2M=./models/SBERT_pegasus__complete_realnewslike_3_MIL_steps/checkpoint-2000000
PRETRAINED_3M=./models/SBERT_pegasus__complete_realnewslike_3_MIL_steps
PRETRAINED_4M=./models/SBERT_pegasus__complete_realnewslike_4_MIL_steps

# Appended to the name of everything an earlier run produced.
VERSION_SUFFIX="__version1"

# Appended, under --eval-only, to the step 5 results of the summaries being
# replaced. Timestamped, so repeated --eval-only runs never collide with each
# other or with "__version1".
STALE_SUFFIX="__before_regen_$(date '+%Y%m%d-%H%M%S')"

# --- fine-tuning ---
FT_MAX_STEPS=100000
FT_TAG=100k                # the "_pt_100k_ft_" part of the model folder names
FT_LR=3e-05
FT_LOGGING_STEP=25000
FT_WARMUP_STEPS=500

# --max_target_length per dataset, used for BOTH fine-tuning and generation.
# 128 matches the PMI / ROUGE runs for both datasets.
CNN_TARGET_LEN=128
WIKIHOW_TARGET_LEN=128

# Every PMI / ROUGE checkpoint is the END of its own pretraining run (max_steps
# equal to the checkpoint, learning rate annealed to ~0). A checkpoint saved in
# the middle of a longer run -- e.g. checkpoint-1000000 of a 3M-step run -- was
# taken at a much higher learning rate, so it is not like for like and the
# script refuses it. Set to 1 (here, or as ALLOW_MID_RUN_CHECKPOINTS=1 in front
# of the command) to use such a checkpoint anyway.
ALLOW_MID_RUN_CHECKPOINTS=${ALLOW_MID_RUN_CHECKPOINTS:-1}

# --- directories ---
MODELS_DIR=./models
FT_MODELS_DIR=./finetuned_models
PRED_DIR=./eval_generated_pred
ANALYSIS_DIR=./evaluation_and_analysis
# Separate from the main pipeline's logs, which already hold the version-1 runs.
LOG_DIR=./pipeline_logs/SBERT_refinetune_cnn_wikihow_128

# --- conda environments ---
# Set any of these to "" to run that stage in whatever environment is already
# active instead of going through `conda run`.
ENV_TRAIN=factP     # fine-tuning, generation
ENV_STEP1=factP         # ROUGE + BERTScore
ENV_STEP2=qaeval       # QAEval
ENV_STEP3=llm_score         # DeBERTa BERTScore

###############################################################################
# per-dataset settings
###############################################################################
#
# Batch size / accumulation pairs are the ones the existing scripts use.

dataset_config() {
    case "$1" in
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
            echo "unknown dataset: $1 (this script only handles cnn and wikihow)" >&2
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

# src/main.py loads the tokenizer from --model_name as well, so a pretrained
# model folder is only usable when the tokenizer was saved next to it.
has_tokenizer() {
    [ -f "$1/tokenizer_config.json" ] && { [ -f "$1/spiece.model" ] || [ -f "$1/tokenizer.json" ]; }
}

# Prints one field ($2, e.g. max_steps) of a model folder's trainer_state.json,
# or nothing when there is no such file.
trainer_state_field() {
    [ -f "$1/trainer_state.json" ] || return 0
    python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' \
        "$1/trainer_state.json" "$2"
}

# "file.ext" -> "file__version1.ext"; a directory or a file without an
# extension just gets the suffix appended. $2 overrides the suffix.
versioned_name() {
    _path="$1"
    _suffix="${2:-$VERSION_SUFFIX}"
    _base=$(basename "$_path")
    _dir=$(dirname "$_path")
    if [ -d "$_path" ]; then
        echo "$_path$_suffix"
        return
    fi
    case "$_base" in
        *.partial.jsonl) echo "$_dir/${_base%.partial.jsonl}$_suffix.partial.jsonl" ;;
        *.*)             echo "$_dir/${_base%.*}$_suffix.${_base##*.}" ;;
        *)               echo "$_path$_suffix" ;;
    esac
}

# Renames one path to its "__version1" name (or the suffix in $2) when it
# exists. Refuses to overwrite an existing one -- that would destroy the only
# copy of the old run.
archive_path() {
    _src="$1"
    [ -e "$_src" ] || return 0

    _dst=$(versioned_name "$_src" "$2")
    if [ -e "$_dst" ]; then
        echo "refusing to overwrite $_dst while archiving $_src" >&2
        echo "move one of them away by hand, then re-run" >&2
        exit 1
    fi

    mv "$_src" "$_dst"
    echo "  $_src"
    echo "    -> $_dst"
}

###############################################################################
# stage 0 -- move the earlier run's outputs aside
###############################################################################
#
# Per checkpoint n and dataset ds, everything that belongs to the old
# (64-token cnn) run:
#
#   finetuned_models/SBERT_pegasus_complete_<n>M_pt_100k_ft_<ds>_comb/
#   eval_generated_pred/eval_results_SBERT_pegasus_complete_<n>M_pt_100k_ft_<ds>_comb/
#       (the whole folder: generated_predictions.txt and whatever do_predict
#        wrote next to it)
#   evaluation_and_analysis/<ds>_result_files/
#       <ds>_combined_results_for_analysis__step{1,2,3}_only_sbert_<n>M.{json,log}
#       sbert_pegasus_<ds>_generated_summaries/generated_predictions_<n>M.txt
#       <ds>_<n>M_sbert_vs_{pmi,rouge}_llm_judge_vs_reference_summaries__step5.{json,log,partial.jsonl}
#
# and once per dataset the step 5 summary logs, which are rebuilt from the
# per-checkpoint files on disk and would otherwise mix the two versions:
#       <ds>_ALL_checkpoints_sbert_vs_{pmi,rouge}_..._step5_summary.log
#
# A marker file makes this run exactly once per checkpoint and dataset, so a
# re-run after a crash does not mistake the NEW outputs for old ones.

archive_version1() {
    n="$1"
    ds="$2"
    dataset_config "$ds"

    marker="$LOG_DIR/.archived_${n}M_${ds}"
    if [ -f "$marker" ]; then
        say "archive ${n}M / ${ds} -- already done, skipping"
        return 0
    fi

    say "archive ${n}M / ${ds}  (old outputs get \"$VERSION_SUFFIX\")"

    res="$ANALYSIS_DIR/$DS_RESULT_FOLDER"

    archive_path "$FT_MODELS_DIR/${KIND}_pegasus_complete_${n}M_pt_${FT_TAG}_ft_${DS_EVAL_SUFFIX}"
    archive_path "$PRED_DIR/eval_results_${KIND}_pegasus_complete_${n}M_pt_${FT_TAG}_ft_${DS_EVAL_SUFFIX}"

    for s in 1 2 3; do
        for ext in json log; do
            archive_path "$res/${ds}_combined_results_for_analysis__step${s}_only_sbert_${n}M.$ext"
        done
    done

    archive_path "$res/$DS_SUMMARY_FOLDER/generated_predictions_${n}M.txt"

    for opponent in pmi rouge; do
        base="$res/${ds}_${n}M_${KIND_LOWER}_vs_${opponent}_llm_judge_vs_reference_summaries__step5"
        archive_path "$base.json"
        archive_path "$base.log"
        archive_path "$base.partial.jsonl"
    done

    dataset_marker="$LOG_DIR/.archived_summaries_${ds}"
    if [ ! -f "$dataset_marker" ]; then
        for opponent in pmi rouge; do
            archive_path "$res/${ds}_ALL_checkpoints_${KIND_LOWER}_vs_${opponent}_llm_judge_vs_reference_summaries__step5_summary.log"
        done
        touch "$dataset_marker"
    fi

    touch "$marker"
}

# --eval-only's counterpart of stage 0, limited to what regeneration makes
# stale: the step 5 judge results of this checkpoint and dataset, which judged
# the summaries about to be replaced. They get STALE_SUFFIX, not "__version1",
# which stage 0 may already have used for the 64-token run. The fine-tuned
# model and the step 1-3 results stay where they are -- generation needs the
# model, and the step 1-3 results are replaced in place by stage 5.
#
# The step 5 summary logs are left alone: step 5 rebuilds them from the
# per-checkpoint files on disk on its next run.
move_aside_stale_step5() {
    n="$1"
    ds="$2"
    dataset_config "$ds"

    res="$ANALYSIS_DIR/$DS_RESULT_FOLDER"
    _moved=""

    for opponent in pmi rouge; do
        base="$res/${ds}_${n}M_${KIND_LOWER}_vs_${opponent}_llm_judge_vs_reference_summaries__step5"
        for f in "$base.json" "$base.log" "$base.partial.jsonl"; do
            if [ -e "$f" ]; then
                [ -n "$_moved" ] || say "step 5 results of ${n}M / ${ds} judged the old summaries -- moving them aside"
                _moved=1
                archive_path "$f" "$STALE_SUFFIX"
            fi
        done
    done
}

###############################################################################
# the pretrained model to start from
###############################################################################
#
# Taken from PRETRAINED_<n>M (see the configuration). The folder must hold a
# model and its tokenizer, must be at n million steps, and -- unless
# ALLOW_MID_RUN_CHECKPOINTS=1 -- come from a run that ENDED at n million steps.

resolve_pretrained() {
    n="$1"
    want_steps="${n}000000"
    eval "PRETRAINED=\${PRETRAINED_${n}M:-}"

    if [ -z "$PRETRAINED" ]; then
        echo "PRETRAINED_${n}M is not set (see the configuration at the top)" >&2
        exit 1
    fi
    if [ ! -d "$PRETRAINED" ]; then
        echo "PRETRAINED_${n}M does not exist: $PRETRAINED" >&2
        exit 1
    fi
    if ! model_is_saved "$PRETRAINED"; then
        echo "PRETRAINED_${n}M holds no pytorch_model.bin / model.safetensors: $PRETRAINED" >&2
        exit 1
    fi
    if ! has_tokenizer "$PRETRAINED"; then
        echo "PRETRAINED_${n}M holds no tokenizer files (tokenizer_config.json +" >&2
        echo "spiece.model or tokenizer.json), which src/main.py needs: $PRETRAINED" >&2
        exit 1
    fi

    # A path pointing at the wrong checkpoint is an easy mistake when editing
    # four of them by hand, and nothing downstream would notice it.
    global_step=$(trainer_state_field "$PRETRAINED" global_step)
    if [ -n "$global_step" ] && [ "$global_step" != "$want_steps" ]; then
        echo "PRETRAINED_${n}M is at step $global_step, not $want_steps: $PRETRAINED" >&2
        exit 1
    fi

    max_steps=$(trainer_state_field "$PRETRAINED" max_steps)
    if [ -n "$max_steps" ] && [ "$max_steps" != "$want_steps" ]; then
        echo "" >&2
        echo "$PRETRAINED was saved at step $want_steps of a run with max_steps=$max_steps," >&2
        echo "so its learning rate had not been annealed yet. Every PMI / ROUGE checkpoint" >&2
        echo "is the end of its own run, so this model is not like for like with them." >&2
        if [ "$ALLOW_MID_RUN_CHECKPOINTS" != "1" ]; then
            echo "Set ALLOW_MID_RUN_CHECKPOINTS=1 to use it anyway." >&2
            exit 1
        fi
        echo "ALLOW_MID_RUN_CHECKPOINTS=1 -- using it anyway." >&2
    elif [ -z "$max_steps" ]; then
        echo "  note: $PRETRAINED has no trainer_state.json, cannot check its step count" >&2
    fi

    echo "  pretrained ${n}M: $PRETRAINED"
}

###############################################################################
# stage 1 -- fine-tuning
###############################################################################

finetune_dataset() {
    n="$1"
    ds="$2"
    dataset_config "$ds"

    out_dir="$FT_MODELS_DIR/${KIND}_pegasus_complete_${n}M_pt_${FT_TAG}_ft_${DS_EVAL_SUFFIX}"

    if model_is_saved "$out_dir"; then
        say "finetune ${n}M / ${ds} -- already done, skipping"
        return 0
    fi

    resolve_pretrained "$n"

    say "finetune ${n}M / ${ds}  (${FT_MAX_STEPS} steps, target length ${DS_TARGET_LEN})"

    # --overwrite_output_dir: save_steps equals max_steps, so a crashed run
    # leaves no checkpoint to resume from -- only a non-empty folder, which
    # src/main.py would otherwise refuse to train into. A finished model never
    # gets here (see the skip above).
    run_stage "$LOG_DIR/${n}M/finetune_${ds}.log" "$ENV_TRAIN" \
        deepspeed --master_port=$PORT --include=localhost:$GPU_IDX src/main.py --fp16 \
        --deepspeed src/ds_config.json \
        --data_dir "finetune_data/$DS_FT_FOLDER" --do_finetune \
        --do_train --model_name "$PRETRAINED" \
        --evaluation_strategy no \
        --per_device_train_batch_size $DS_TRAIN_BS --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps $DS_GRAD_ACC \
        --learning_rate $FT_LR --weight_decay 0.01 --label_smoothing 0.1 \
        --max_source_length 512 --max_target_length $DS_TARGET_LEN \
        --logging_step $FT_LOGGING_STEP --max_steps $FT_MAX_STEPS \
        --warmup_steps $FT_WARMUP_STEPS --save_steps $FT_MAX_STEPS \
        --output_dir "$out_dir" --overwrite_output_dir \
        --contrastive_learning --pertubation_type intrinsic --num_negatives 5 --contrastive_weight 5
}

###############################################################################
# stage 2 -- generation
###############################################################################

generate_predictions() {
    n="$1"
    ds="$2"
    dataset_config "$ds"

    ft_model="$FT_MODELS_DIR/${KIND}_pegasus_complete_${n}M_pt_${FT_TAG}_ft_${DS_EVAL_SUFFIX}"
    out_dir="$PRED_DIR/eval_results_${KIND}_pegasus_complete_${n}M_pt_${FT_TAG}_ft_${DS_EVAL_SUFFIX}"

    # --eval-only exists to regenerate, so it never takes this skip.
    if [ "$EVAL_ONLY" != "1" ] && [ -s "$out_dir/generated_predictions.txt" ]; then
        say "generate ${n}M / ${ds} -- already done, skipping"
        return 0
    fi

    say "generate ${n}M / ${ds}  (target length ${DS_TARGET_LEN})"

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

    # The failure this script exists to fix, checked on the new output: the
    # share of summaries that do not end like a sentence (PMI's cnn: ~1%).
    python3 - "$out_dir/generated_predictions.txt" "$ds" "$n" <<'PYEOF'
import sys
lines = [l.rstrip("\n") for l in open(sys.argv[1], encoding="utf-8")]
cut = sum(1 for l in lines if not l.rstrip().endswith((".", "!", "?", '"', "'", ")")))
print(f"  {sys.argv[2]} {sys.argv[3]}M: {len(lines)} summaries, "
      f"{cut / len(lines) * 100:.1f}% without sentence-final punctuation")
PYEOF
}

###############################################################################
# stage 3 -- put the predictions where steps 1-3 look for them
###############################################################################

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
# stage 4 -- the three analysis scripts, cnn and wikihow only
###############################################################################
#
# The step scripts run xsum, cnn and wikihow from a hard-coded __main__, so
# they are imported instead and only the functions for DATASETS are called.
# Importing still runs their module-level setup (flags, model preloading),
# which is what their functions rely on. They resolve every path relative to
# their own folder, so they run from inside it; LOG_DIR is made absolute up
# front so the logs still land in the right place.

step_call() {
    # $1 = module, $2 = function prefix -> "import m; m.<prefix>cnn(); m.<prefix>wikihow()"
    _code="import $1"
    for ds in $DATASETS; do
        _code="$_code; $1.$2$ds()"
    done
    echo "$_code"
}

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
# stage 5 -- rename the results so the checkpoint is part of the name
###############################################################################

# On a normal run the old results were renamed to "__version1" in stage 0, so
# each destination here is free. Under --analysis-only and --eval-only they were
# not, and `mv` replaces them in place -- which is what those modes are for, so
# it is reported rather than refused.
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
                _note=""
                [ -f "${base}_${n}M.$ext" ] && _note="   (replaced)"
                mv "$base.$ext" "${base}_${n}M.$ext"
                echo "  ${ds}_combined_results_for_analysis__step${s}_only_sbert_${n}M.$ext$_note"
            done
        done
    done
}

# True once every step3 result of this checkpoint has been archived. The old
# ones were renamed to "__version1" in stage 0, so what is found here is new.
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
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        -*)
            echo "unknown option: $1" >&2
            echo "usage: $0 [--eval-only | --analysis-only] [checkpoint ...]" >&2
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

for n in $CHECKPOINTS; do
    case "$n" in
        1|2|3|4) ;;
        *) echo "checkpoint must be one of 1 2 3 4, got: $n" >&2; exit 1 ;;
    esac
done

# Checked for every requested checkpoint up front, so a wrong path or a missing
# input fails in seconds rather than after the earlier checkpoints have run.
# --analysis-only and --eval-only never fine-tune, so they want the summaries
# to score, or the fine-tuned models to generate from, rather than the
# pretrained models to start from.
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
            # Stage 0 leaves this marker once it has moved the 64-token model
            # aside. Without it, the model found above may still be that old one
            # -- or it was re-fine-tuned on another machine, which is fine.
            if [ ! -f "$LOG_DIR/.archived_${n}M_${ds}" ]; then
                echo "    note: no record of this script re-fine-tuning ${n}M / ${ds} here --" >&2
                echo "    if it is still the original (pre-re-fine-tune) model, run" >&2
                echo "    without --eval-only instead" >&2
            fi
        done
    done
elif [ "$ANALYSIS_ONLY" = "1" ]; then
    for n in $CHECKPOINTS; do
        for ds in $DATASETS; do
            dataset_config "$ds"
            preds="$PRED_DIR/eval_results_${KIND}_pegasus_complete_${n}M_pt_${FT_TAG}_ft_${DS_EVAL_SUFFIX}/generated_predictions.txt"
            if [ ! -s "$preds" ]; then
                echo "--analysis-only needs summaries that are already there, but" >&2
                echo "$preds is missing or empty." >&2
                echo "Run without --analysis-only to fine-tune and generate them first." >&2
                exit 1
            fi
            echo "  ${n}M / ${ds}: $preds"
        done
    done
else
    for n in $CHECKPOINTS; do
        resolve_pretrained "$n"
    done
fi

mkdir -p "$LOG_DIR"
LOG_DIR=$(cd "$LOG_DIR" && pwd)

if [ "$ANALYSIS_ONLY" = "1" ]; then
    say "SBERT steps 1-3 only -- checkpoints: $CHECKPOINTS"
    echo "  datasets         : $DATASETS"
    echo "  fine-tune        : skipped (--analysis-only)"
    echo "  generation       : skipped (--analysis-only)"
    echo "  step 1-3 results : replaced in place, no \"$VERSION_SUFFIX\" copy kept"
    echo "  logs             : $LOG_DIR"
elif [ "$EVAL_ONLY" = "1" ]; then
    say "SBERT generation + steps 1-3 -- checkpoints: $CHECKPOINTS"
    echo "  datasets         : $DATASETS"
    echo "  target lengths   : cnn=$CNN_TARGET_LEN wikihow=$WIKIHOW_TARGET_LEN"
    echo "  archive / ft     : skipped (--eval-only)"
    echo "  generation       : rerun from finetuned_models, summaries replaced in place"
    echo "  step 1-3 results : replaced in place, no \"$VERSION_SUFFIX\" copy kept"
    echo "  step 5 results   : moved aside with \"$STALE_SUFFIX\""
    echo "  logs             : $LOG_DIR"
else
    say "SBERT re-fine-tune -- checkpoints: $CHECKPOINTS"
    echo "  datasets         : $DATASETS"
    echo "  target lengths   : cnn=$CNN_TARGET_LEN wikihow=$WIKIHOW_TARGET_LEN"
    echo "  old outputs      : renamed with \"$VERSION_SUFFIX\""
    echo "  logs             : $LOG_DIR"
fi

for n in $CHECKPOINTS; do
    mkdir -p "$LOG_DIR/${n}M"

    if [ "$EVAL_ONLY" = "1" ]; then
        for ds in $DATASETS; do
            move_aside_stale_step5 "$n" "$ds"
            generate_predictions "$n" "$ds"
        done
    elif [ "$ANALYSIS_ONLY" != "1" ]; then
        for ds in $DATASETS; do
            archive_version1 "$n" "$ds"
            finetune_dataset "$n" "$ds"
            generate_predictions "$n" "$ds"
        done
    fi

    # Under --analysis-only and --eval-only the results that are already there
    # are precisely the ones being replaced, so the skip does not apply.
    if [ "$ANALYSIS_ONLY" != "1" ] && [ "$EVAL_ONLY" != "1" ] && analysis_already_done "$n"; then
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

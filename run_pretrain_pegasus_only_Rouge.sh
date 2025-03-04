#!/bin/sh

GPU_IDX=0

deepspeed --include=localhost:"$GPU_IDX" src/main.py --fp16 \
--data_dir ./c4_realnewslike_processed_rouge_combined \
--do_train --do_pretrain --model_name facebook/bart-base \
--deepspeed src/ds_config.json \
--per_device_train_batch_size 16 --gradient_accumulation_steps 2 \
--learning_rate 1e-4 --weight_decay 0.01 \
--logging_step 5000  --max_steps 500000 \
--warmup_steps 20000 --save_steps 40000 \
--max_source_length 512 --max_target_length 256 \
--output_dir ./models/rouge_pegasus_1_mil_subset_500000_steps --pretrain_model_type bart_base --tokenize_on_fly

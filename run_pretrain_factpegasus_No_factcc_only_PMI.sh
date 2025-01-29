#!/bin/sh

GPU_IDX=0

deepspeed --include=localhost:"$GPU_IDX" src/main.py --fp16 \
--data_dir ./c4_realnewslike_processed_combined_No_factcc_only_PMI \
--do_train --do_pretrain --model_name facebook/bart-base \
--deepspeed src/ds_config.json \
--per_device_train_batch_size 16 --gradient_accumulation_steps 2 \
--learning_rate 1e-4 --weight_decay 0.01 \
--logging_step 100  --max_steps 500 \
--warmup_steps 20000 --save_steps 5000 \
--max_source_length 512 --max_target_length 256 \
--output_dir ./models/factpegasus_No_factcc_only_PMI_500_steps --pretrain_model_type bart_base --tokenize_on_fly


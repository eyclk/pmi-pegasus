#!/bin/sh

GPU_IDX=0

deepspeed --include=localhost:"$GPU_IDX" src/main.py --fp16 \
--data_dir ./PREPROCESSED_DATASETS/COMPLETE_c4_realnewslike_processed_PMI \
--do_train --do_pretrain --model_name facebook/bart-base --resume_from_checkpoint ./models/PMI_pegasus__complete_realnewslike_7_MIL_steps/checkpoint-7000000 \
--deepspeed src/ds_config.json \
--per_device_train_batch_size 16 --gradient_accumulation_steps 2 \
--learning_rate 1e-4 --weight_decay 0.01 \
--logging_step 20000  --max_steps 8000000 \
--warmup_steps 20000 --save_steps 500000 \
--max_source_length 512 --max_target_length 256 \
--output_dir ./models/PMI_pegasus__complete_realnewslike_8_MIL_steps --pretrain_model_type bart_base --tokenize_on_fly





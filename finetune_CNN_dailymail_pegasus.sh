#!/bin/sh

GPU_IDX=0

PORT="29501"

deepspeed --master_port=$PORT --include=localhost:$GPU_IDX src/main.py --fp16 \
--deepspeed src/ds_config.json \
--data_dir finetune_data/cnn_dailymail_comb --do_finetune \
--do_train --model_name models/rouge_pegasus_1_mil_subset_750000_steps \
--evaluation_strategy no \
--per_device_train_batch_size 16 --per_device_eval_batch_size 8 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-05 --weight_decay 0.01 --label_smoothing 0.1 \
--max_source_length 512 --max_target_length 64 \
--logging_step 5000 --max_steps 250000 \
--warmup_steps 500 --save_steps 100000 \
--output_dir ./finetuned_models/rouge_pegasus_ft_cnn_dailymail_comb \
--contrastive_learning --pertubation_type intrinsic --num_negatives 5 --contrastive_weight 5


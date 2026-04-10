#!/bin/sh

GPU_IDX=0

PORT="29501"

deepspeed --master_port=$PORT --include=localhost:$GPU_IDX src/main.py --fp16 \
--deepspeed src/ds_config.json \
--data_dir finetune_data/wikihow_comb --do_finetune \
--do_train --model_name models/PMI_pegasus__complete_realnewslike_4_MIL_steps  \
--evaluation_strategy no \
--per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
--gradient_accumulation_steps 4 \
--learning_rate 3e-05 --weight_decay 0.01 --label_smoothing 0.1 \
--max_source_length 512 --max_target_length 128 \
--logging_step 25000 --max_steps 100000 \
--warmup_steps 500 --save_steps 100000 \
--output_dir ./finetuned_models/PMI_pegasus_complete_4M_pt_100k_ft_wikihow_comb \
--contrastive_learning --pertubation_type intrinsic --num_negatives 5 --contrastive_weight 5




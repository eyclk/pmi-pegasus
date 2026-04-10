#!/bin/sh

python src/main.py --fp16 \
--data_dir finetune_data/wikihow_comb --do_predict --predict_with_generate \
--model_name ./finetuned_models/PMI_pegasus_complete_4M_pt_100k_ft_wikihow_comb \
--per_device_train_batch_size 16 --per_device_eval_batch_size 8 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-05 --weight_decay 0.01 --label_smoothing 0.1 \
--max_source_length 512 --max_target_length 128 \
--logging_step 1000  \
--output_dir ./eval_generated_pred/eval_results_PMI_pegasus_complete_4M_pt_100k_ft_wikihow_comb \




#!/bin/sh

python src/main.py --fp16 \
--data_dir finetune_data/cnn_dailymail_comb --do_predict --predict_with_generate \
--model_name ./finetuned_models/rouge_pegasus_ft_cnn_dailymail_comb \
--per_device_train_batch_size 16 --per_device_eval_batch_size 8 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-05 --weight_decay 0.01 --label_smoothing 0.1 \
--max_source_length 512 --max_target_length 64 \
--logging_step 1000  \
--output_dir ./eval_extras/eval_results_rouge_pegasus_ft_cnn_dailymail_comb \






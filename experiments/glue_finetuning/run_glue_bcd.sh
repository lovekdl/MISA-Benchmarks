# roberta-base
CUDA_VISIBLE_DEVICES=2
python -u experiments/glue_finetuning/run_glue.py \
    --model_name_or_path roberta-base\
    --bcd_update_order grad_distribution\
    --task_name copa \
    --do_train True \
    --do_eval True\
    --do_predict False \
    --optimizer bcd-optimizer \
    --bcd_activated_layers 1 \
    --bcd_interval_steps 30 \
    --per_device_train_batch_size 16 \
    --seed 42 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine\
    --max_seq_length 512 \
    --logging_steps 1 \
    --save_total_limit 0 \
    --evaluation_strategy epoch \
    --save_strategy no \
    --report_to none \
    --run_name copa \
    --overwrite_output_dir \
    --output_dir result/lora/test  \
    --report_to wandb \
    --pad_to_max_length False \

export SEED=1234
export MODEL=llama_350m
export LEARNING_RATE=1e-3
export BCD_INTERVAL=50
export BANDIT_ETA=1.0
export BCD_ORDER=bandit_sq
python -m torch.distributed.run --standalone --nproc_per_node=1 experiments/c4_pretraining/run_llama_pretraining.py \
    --model_config experiments/c4_pretraining/configs/$MODEL.json \
    --single_gpu \
    --max_length 256 \
    --dtype bfloat16 \
    --num_training_steps 55000 \
    --warmup_steps 5500 \
    --eval_every 1500 \
    --save_every 10000 \
    --total_batch_size 256 \
    --batch_size 32 \
    --save_dir experiments/c4_pretraining/results/MISA \
    --seed $SEED \
    --optimizer bcd-optimizer \
    --lr $LEARNING_RATE \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --weight_decay 0.0 \
    --grad_clipping 0.0 \
    --bcd_activated_layers 1\
    --bcd_interval_steps $BCD_INTERVAL\
    --bcd_update_order $BCD_ORDER\
    --granularity module\
    --param_ratio_limit 0.03\
    --bandit_eta $BANDIT_ETA\
    --include_embedding_and_lm_head \
    --wandb_project LDAdam_C4_${MODEL} \
    --wandb_run_name MISA_${SEED}_LR${LEARNING_RATE}_Interval${BCD_INTERVAL}_ETA${BANDIT_ETA}_Order[${BCD_ORDER}]
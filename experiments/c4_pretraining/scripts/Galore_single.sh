export SEED=1234
export MODEL=llama_350m
export LEARNING_RATE=1e-3
export GALORE_RANK=256
python -m torch.distributed.run --standalone --nproc_per_node=1 experiments/c4_pretraining/run_llama_pretraining.py \
    --model_config experiments/c4_pretraining/configs/$MODEL.json \
    --max_length 256 \
    --dtype bfloat16 \
    --single_gpu \
    --num_training_steps 55000 \
    --warmup_steps 5500 \
    --eval_every 1500 \
    --save_every 10000 \
    --total_batch_size 256 \
    --batch_size 32 \
    --save_dir experiments/c4_pretraining/GALORE/RANK${GALORE_RANK} \
    --seed $SEED \
    --optimizer galore \
    --lr $LEARNING_RATE \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --weight_decay 0.0 \
    --grad_clipping 0.0 \
    --wandb_project LDAdam_C4_${MODEL} \
    --wandb_run_name GALORE_Single_[${GALORE_RANK}]_${SEED}_LR${LEARNING_RATE}
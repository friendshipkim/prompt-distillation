ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
        train.py \
        recipes/zephyr-7b-beta/sft/distill_width.yaml \
    --report_to=wandb
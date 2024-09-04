ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
        train.py \
        recipes/zephyr-7b-beta/sft/config_full.yaml \
    --report_to=wandb
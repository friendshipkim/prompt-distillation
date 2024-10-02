ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
        train_sft.py \
        recipes/zephyr-7b-beta/sft/student_width.yaml \
    --report_to=wandb
# ACCELERATE_LOG_LEVEL=info accelerate launch \
#     --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
#         train_sft.py \
#         recipes/zephyr-7b-beta/sft/student.yaml \
#     --report_to=wandb

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
        train_sft.py \
        recipes/zephyr-7b-beta/sft/teacher.yaml \
    --report_to=wandb
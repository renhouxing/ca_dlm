myrun() {
    python /mnt/cache/code/scripts/srun.py "${@:1}"
}

mywait() {
    python /mnt/cache/code/scripts/wait.py "${@:1}"
}

### Train

# --do_eval --eval_strategy steps --eval_steps 100 --save_total_limit 5 --per_device_eval_batch_size 1 --load_best_model_at_end 

# myrun -j train -e dlm -n 2 -c "HF_DATASETS_CACHE=/mnt/cache/code/.cache/huggingface/datasets torchrun --node_rank \${RANK} --master_addr \${MASTER_ADDR} --master_port \${MASTER_PORT} --nnodes \${WORLD_SIZE} --nproc_per_node 8 train.py --seed 42 --report_to tensorboard --dataloader_num_workers 8 --remove_unused_columns False --save_steps 1000 --max_len 2048 --warmup_steps 100 --logging_steps 10 --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs \"{\\\"min_lr\\\": 5e-06}\" --group_by_length --bf16 --do_train --learning_rate 1e-5 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --deepspeed config/stage_1.json --model_cfg data/Dream-7B-Instruct --train_file data/dream_sft.jsonl --output_dir runs/Dream-Lora --attn_layer 0 --lora"

# for coef in 1e-02; do
#     for alg in right wrong; do
#         myrun -j train -e dlm -n 2 -c "HF_DATASETS_CACHE=/mnt/cache/code/.cache/huggingface/datasets torchrun --node_rank \${RANK} --master_addr \${MASTER_ADDR} --master_port \${MASTER_PORT} --nnodes \${WORLD_SIZE} --nproc_per_node 8 train.py --seed 42 --report_to tensorboard --dataloader_num_workers 8 --remove_unused_columns False --save_steps 1000 --max_len 1024 --warmup_steps 100 --logging_steps 10 --lr_scheduler_type constant_with_warmup --group_by_length --bf16 --do_train --learning_rate 1e-5 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --deepspeed config/stage_1.json --model_cfg data/Dream-7B-Instruct --train_file data/dream_sft.jsonl --output_dir runs/Dream-Lora --attn_alg ${alg} --attn_coef ${coef} --lora --attn_layer 28"
#     done
# done

# myrun -j train -e dlm -n 2 -c "HF_DATASETS_CACHE=/mnt/cache/code/.cache/huggingface/datasets torchrun --node_rank \${RANK} --master_addr \${MASTER_ADDR} --master_port \${MASTER_PORT} --nnodes \${WORLD_SIZE} --nproc_per_node 8 train.py --seed 42 --report_to tensorboard --dataloader_num_workers 8 --remove_unused_columns False --save_steps 1000 --max_len 1024 --warmup_steps 100 --logging_steps 10 --lr_scheduler_type constant_with_warmup --group_by_length --bf16 --do_train --learning_rate 1e-05 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --deepspeed config/stage_1.json --model_cfg data/LLaDA-8B-Instruct --train_file data/dream_sft.jsonl --output_dir runs/LLaDA-Lora-Full --lora"

for coef in 1e-04 5e-04; do
    myrun -j train -e dlm -n 2 -c "HF_DATASETS_CACHE=/mnt/cache/code/.cache/huggingface/datasets torchrun --node_rank \${RANK} --master_addr \${MASTER_ADDR} --master_port \${MASTER_PORT} --nnodes \${WORLD_SIZE} --nproc_per_node 8 train.py --seed 42 --report_to tensorboard --dataloader_num_workers 8 --remove_unused_columns False --save_steps 1000 --max_len 2048 --warmup_steps 100 --logging_steps 10 --lr_scheduler_type constant_with_warmup --group_by_length --bf16 --do_train --learning_rate 1e-05 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --gradient_checkpointing --deepspeed config/stage_1.json --model_cfg data/LLaDA-8B-Instruct --train_file data/deepseek_0528.jsonl --output_dir runs/LLaDA-Lora-Full-0 --attn_alg all --attn_coef ${coef} --lora"
done 
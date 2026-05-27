#!/usr/bin/env bash

myrun() {
    python /mnt/cache/code/scripts/srun.py "${@:1}"
}

DIRS=(
    "LLaDA-Lora-Full-New_all_1e-06"
    "LLaDA-Lora-Full-New_all_1e-04"
    "LLaDA-Lora-Full-New_all_1e-05"
)

STATE_DIR="/tmp/dir-watch-once"
rm -rf "$STATE_DIR"
mkdir -p "$STATE_DIR"

run_commands() {
    local dir="$1"

    echo "Eval $dir"

    for step in 4 2; do
        for task in humaneval_instruct mbpp_instruct gsm8k_cot minerva_math mmlu_pro gpqa_main_cot_zeroshot ifeval; do
            case "$task" in
                humaneval_instruct|mbpp_instruct)
                max_tokens=512
                other_args=""
                ;;

                gsm8k_cot|minerva_math|ifeval)
                max_tokens=512
                other_args="HF_HOME=/mnt/cache/code/.cache/huggingface"
                ;;

                mmlu_pro|gpqa_main_cot_zeroshot)
                max_tokens=128
                other_args="HF_HOME=/mnt/cache/code/.cache/huggingface"
                ;;
            esac
        
            myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; pred_per_step=${step}; lora=${dir}; score_mode=attnAlignLogit; ${other_args} accelerate launch test_llada.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_llada/\${lora}/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/LLaDA-8B-Instruct,lora_path=runs/\${lora}/checkpoint-final,generation_len=${max_tokens},pred_per_step=\${pred_per_step},score_mode=\${score_mode}"
        done
    done
}

while true; do
    all_done=true

    for dir in "${DIRS[@]}"; do
        mark="$STATE_DIR/$(echo "$dir" | sed 's#[/: ]#_#g').done"

        if [[ -f "$mark" ]]; then
        continue
        fi

        all_done=false

        if [[ -d "runs/${dir}/checkpoint-final" ]]; then
        run_commands "$dir"
        touch "$mark"
        fi
    done

    if [[ "$all_done" == true ]]; then
        exit 0
    fi

    sleep 30
done
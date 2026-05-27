
myrun() {
    python /mnt/cache/code/scripts/srun.py "${@:1}"
}

# https://github.com/EleutherAI/lm-evaluation-harness/pull/3710

# myrun -g 8 -e dlm -j test_humaneval_instruct -c "tasks=humaneval_instruct; lora_path=Dream-Lora-Full_all_1e-02; pred_per_step=112; score_mode=attnAlign; accelerate launch test_fix_step.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_dynamic/\${lora_path}/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=runs/\${lora_path}/checkpoint-final,generation_len=256,pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

# myrun -g 8 -e dlm -j test_mbpp_instruct -c "tasks=mbpp_instruct; lora_path=Dream-Lora-Full_all_1e-02; pred_per_step=50; score_mode=attnAlign; accelerate launch test_fix_step.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_dynamic/\${lora_path}/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=runs/\${lora_path}/checkpoint-final,generation_len=256,pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

# myrun -g 8 -e dlm -j test_gsm8k_cot -c "tasks=gsm8k_cot; lora_path=Dream-Lora-Full_all_1e-02; pred_per_step=70; score_mode=attnAlign; ${other_args} accelerate launch test_fix_step.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_dynamic/\${lora_path}/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=runs/\${lora_path}/checkpoint-final,generation_len=256,pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

# myrun -g 8 -e dlm -j test_minerva_math500 -c "tasks=minerva_math500; lora_path=Dream-Lora-Full_all_1e-02; pred_per_step=60; score_mode=attnAlign; ${other_args} accelerate launch test_fix_step.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_dynamic/\${lora_path}/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=runs/\${lora_path}/checkpoint-final,generation_len=512,pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

# for task in humaneval_instruct mbpp_instruct gsm8k_cot minerva_math500; do
#     case "$task" in
#         humaneval_instruct|mbpp_instruct)
#         max_tokens=256
#         ;;

#         gsm8k_cot|minerva_math500)
#         max_tokens=512
#         ;;
#     esac
#     myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; lora_path=None; pred_per_step=4; score_mode=attnAlignLogit; ${other_args} accelerate launch test_eb.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_eb_llada/\${lora_path}/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/LLaDA-8B-Instruct,lora_path=None,generation_len=${max_tokens},pred_per_step=\${pred_per_step},score_mode=\${score_mode},gamma=0.01"
# done

# humaneval_instruct mbpp_instruct gsm8k_cot minerva_math mmlu_pro gpqa_main_cot_zeroshot ifeval

for step in 8; do
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
    
    
        # =====================================
        # Baseline
        # =====================================

        # myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; pred_per_step=${step}; score_mode=confidence; ${other_args} accelerate launch test.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_dream/Base/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=None,generation_len=${max_tokens},temperature=0.1,top_p=0.9,pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

        # myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; pred_per_step=${step}; score_mode=uniform; ${other_args} accelerate launch test.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_dream/Base/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=None,generation_len=${max_tokens},temperature=0.1,top_p=0.9,pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

        # myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; pred_per_step=${step}; score_mode=attn; ${other_args} accelerate launch test.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_dream/Base/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=None,generation_len=${max_tokens},temperature=0.1,top_p=0.9,pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

        # myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; pred_per_step=${step}; score_mode=entropy; ${other_args} accelerate launch test.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_dream/Base/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=None,generation_len=${max_tokens},temperature=0.1,top_p=0.9,pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

        # myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; pred_per_step=${step}; score_mode=attnAlign; ${other_args} accelerate launch test.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_dream/Base/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=None,generation_len=${max_tokens},pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

        # =====================================
        # Main Methods
        # =====================================

        # myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; lora_path=Dream-Lora-Full_all_1e-02; pred_per_step=${step}; score_mode=attnAlign; ${other_args} accelerate launch test.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_dream/\${lora_path}/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=runs/\${lora_path}/checkpoint-final,generation_len=${max_tokens},pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

        # =====================================
        # Sensitivity Analysis
        # =====================================

        # myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; lora_path=Dream-Lora-Full_all_2e-02; pred_per_step=${step}; score_mode=attnAlign; ${other_args} accelerate launch test.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_dream/\${lora_path}/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=runs/\${lora_path}/checkpoint-final,generation_len=${max_tokens},pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

        # myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; lora_path=Dream-Lora-Full_all_5e-03; pred_per_step=${step}; score_mode=attnAlign; ${other_args} accelerate launch test.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_dream/\${lora_path}/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=runs/\${lora_path}/checkpoint-final,generation_len=${max_tokens},pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

        # =====================================
        # Ablation Study
        # =====================================

        # myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; lora_path=Dream-Lora-1024_right_1e-02; pred_per_step=${step}; score_mode=attnAlign; ${other_args} accelerate launch test.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_dream/\${lora_path}/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=runs/\${lora_path}/checkpoint-final,generation_len=${max_tokens},pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

        # myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; lora_path=Dream-Lora-1024_wrong_1e-02; pred_per_step=${step}; score_mode=attnAlign; ${other_args} accelerate launch test.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_dream/\${lora_path}/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=runs/\${lora_path}/checkpoint-final,generation_len=${max_tokens},pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

        # myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; lora_path=Dream-Lora; pred_per_step=${step}; score_mode=attnAlign; ${other_args} accelerate launch test.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_dream/\${lora_path}/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=runs/\${lora_path}/checkpoint-final,generation_len=${max_tokens},pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

        # myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; lora_path=Dream-Lora; pred_per_step=${step}; score_mode=entropy; ${other_args} accelerate launch test.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_dream/\${lora_path}/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=runs/\${lora_path}/checkpoint-final,temperature=0.1,top_p=0.9,generation_len=${max_tokens},pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

        # =====================================
        # LLaDA
        # =====================================

        # myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; pred_per_step=${step}; score_mode=confidence; ${other_args} accelerate launch test.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_llada/Base/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/LLaDA-8B-Instruct,lora_path=None,generation_len=${max_tokens},pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

        # myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; pred_per_step=${step}; score_mode=attnAlignLogit; ${other_args} accelerate launch test_llada.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_llada/Base/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/LLaDA-8B-Instruct,lora_path=None,generation_len=${max_tokens},pred_per_step=\${pred_per_step},score_mode=\${score_mode}"


        myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; pred_per_step=${step}; lora=LLaDA-Lora-Full-New_all_1e-04; score_mode=attnAlignLogit; ${other_args} accelerate launch test_llada.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_llada/\${lora}/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/LLaDA-8B-Instruct,lora_path=runs/\${lora}/checkpoint-final,generation_len=${max_tokens},pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

        myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; pred_per_step=${step}; lora=LLaDA-Lora-Full-New_all_1e-05; score_mode=attnAlignLogit; ${other_args} accelerate launch test_llada.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_llada/\${lora}/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/LLaDA-8B-Instruct,lora_path=runs/\${lora}/checkpoint-final,generation_len=${max_tokens},pred_per_step=\${pred_per_step},score_mode=\${score_mode}"

        myrun -g 8 -e dlm -j test_${task} -c "tasks=${task}; pred_per_step=${step}; lora=LLaDA-Lora-Full-New_all_1e-06; score_mode=attnAlignLogit; ${other_args} accelerate launch test_llada.py --model attn_dlm --num_fewshot 0 --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks \${tasks} --output_path evals_results_llada/\${lora}/\${tasks}_\${pred_per_step}_\${score_mode} --model_args pretrained=data/LLaDA-8B-Instruct,lora_path=runs/\${lora}/checkpoint-final,generation_len=${max_tokens},pred_per_step=\${pred_per_step},score_mode=\${score_mode}"
    done
done

# tasks=mbpp_instruct; lora_path=LLaDA-Lora-Full_all_1e-02; max_tokens=512; pred_per_step=8; score_mode=attnAlign; ${other_args} accelerate launch --num_processes=1 test.py --model attn_dlm --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks ${tasks} --output_path evals_results_debug/${lora_path}/${tasks}_${pred_per_step}_${score_mode} --model_args pretrained=data/LLaDA-8B-Instruct,lora_path=runs/${lora_path}/checkpoint-final,generation_len=${max_tokens},pred_per_step=${pred_per_step},score_mode=${score_mode} --limit 1

# tasks=gpqa_main_cot_zeroshot; lora_path=Dream-Lora-Full_all_1e-02; max_tokens=512; pred_per_step=8; score_mode=attnAlign; ${other_args} accelerate launch --num_processes=1 test.py --model attn_dlm --batch_size 1 --log_samples --apply_chat_template --confirm_run_unsafe_code --tasks ${tasks} --output_path evals_results_debug/${lora_path}/${tasks}_${pred_per_step}_${score_mode} --model_args pretrained=data/Dream-7B-Instruct,lora_path=runs/${lora_path}/checkpoint-final,generation_len=${max_tokens},pred_per_step=${pred_per_step},score_mode=${score_mode} --limit 1

# python score.py -f Base Dream-Lora-Full_all_1e-02 Dream-Lora
# python score.py -f Dream-Lora-Full_all_1e-02 Dream-Lora-Full_all_2e-02 Dream-Lora-Full_all_5e-03 Dream-Lora-1024_right_1e-02 Dream-Lora-1024_wrong_1e-02
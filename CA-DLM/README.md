## Model Preparation

To output attention weights correctly, please download the model checkpoint from:

```text
Dream-org/Dream-v0-Instruct-7B
```

Then move the `.safetensors` files directly to:

```text
data/Dream-7B-Instruct
```

Please use this local directory instead of the original Hugging Face checkpoint path.

## Training

### Data Format

The training data should be a JSONL file, where each line follows the chat-style format below:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "You are given a list of `n` tasks, each represented as a tuple `(start, end)`, indicating the start and end times of the task. The tasks are sorted by their start times. Your goal is to determine the maximum number of non-overlapping tasks that can be selected. Two tasks are considered non-overlapping if the start time of one task is greater than or equal to the end time of the other.\n\n**Input:**\n- An integer `n` representing the number of tasks.\n- A list of `n` tuples, where each tuple `(start, end)` represents the start and end times of a task.\n\n**Output:**\n- An integer representing the maximum number of non-overlapping tasks that can be selected.\n\n**Constraints:**\n- `1 <= n <= 10^5`\n- `0 <= start < end <= 10^9`\n\n**Sample Input:**\n```python\n3\n1 3\n2 5\n4 6\n```\n\n**Sample Output:**\n```python\n2\n```\n\nPlease enclose your code within delimiters as follows:\n```python\n# YOUR CODE HERE\n```"
    },
    {
      "role": "assistant",
      "content": "I'll solve this step-by-step.\n\n## Problem Understanding\n\nThis is a classic \"Activity Selection Problem\" where we want to find the maximum number of non-overlapping tasks. The key insight is that we should use a greedy approach: always pick the task that finishes earliest among the remaining tasks.\n\n## Step-by-Step Solution\n\n### Step 1: Analyze the Problem\n- We have tasks sorted by start time\n- We want to maximize the number of non-overlapping tasks\n- Two tasks don't overlap if the start time of one >= end time of the other\n\n### Step 2: Algorithm Design\nThe optimal greedy strategy is:\n1. Always pick the task that finishes earliest\n2. This eliminates the maximum possible time for future tasks\n3. Continue until no more tasks can be selected\n\n### Step 3: Implementation Strategy\n1. Initialize counter and track the end time of the last selected task\n2. Iterate through tasks\n3. For each task, if its start time >= last selected task's end time, select it\n4. Update the end time and increment counter\n\n### Step 4: Code Implementation\n\n```python\n# YOUR CODE HERE\nn = int(input())\ntasks = []\n\nfor _ in range(n):\n    start, end = map(int, input().split())\n    tasks.append((start, end))\n\nif n == 0:\n    print(0)\n    exit()\n\ncount = 1\nlast_end = tasks[0][1]\n\nfor i in range(1, n):\n    start, end = tasks[i]\n    if start >= last_end:\n        count += 1\n        last_end = end\n\nprint(count)\n```\n\nThe algorithm runs in O(n) time."
    }
  ]
}
```

## Fine-tuning

Run the following command to fine-tune the model with the attention-calibration objective:

```bash
torchrun --nproc_per_node 8 train.py \
  --seed 42 \
  --report_to tensorboard \
  --dataloader_num_workers 8 \
  --remove_unused_columns False \
  --save_steps 1000 \
  --max_len 1024 \
  --warmup_steps 100 \
  --logging_steps 10 \
  --lr_scheduler_type constant_with_warmup \
  --group_by_length \
  --bf16 \
  --do_train \
  --learning_rate 1e-05 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 64 \
  --deepspeed config/stage_1.json \
  --model_cfg data/Dream-7B-Instruct \
  --train_file <your_data_path> \
  --output_dir runs/Dream-Lora-CA \
  --attn_alg all \
  --attn_coef 0.01 \
  --lora
```

## Evaluation

We use `lm-eval` style evaluation to test the fine-tuned model.

### Evaluation Command

```bash
tasks=humaneval_instruct/mbpp_instruct/gsm8k_cot/minerva_math/mmlu_pro/gpqa_main_cot_zeroshot/ifeval
lora_path=Dream-Lora-CA
pred_per_step=2/4/8
score_mode=attnAlign
max_tokens=512

accelerate launch test.py \
  --model attn_dlm \
  --num_fewshot 0 \
  --batch_size 1 \
  --log_samples \
  --apply_chat_template \
  --confirm_run_unsafe_code \
  --tasks ${tasks} \
  --output_path evals_results_dream/${lora_path}/${tasks}_${pred_per_step}_${score_mode} \
  --model_args pretrained=data/Dream-7B-Instruct,lora_path=runs/${lora_path}/checkpoint-final,generation_len=${max_tokens},pred_per_step=${pred_per_step},score_mode=${score_mode}
```
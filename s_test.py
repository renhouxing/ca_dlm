import numpy as np
import json

def load_jsonl(file_path):
    if isinstance(file_path, list):
        data = []
        for path in file_path:
            with open(path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
    else:   
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    return data

import re
from datetime import datetime
from pathlib import Path


import re
from datetime import datetime
from pathlib import Path


def get_latest_json_paths(directory: str) -> list[Path]:
    directory_path = Path(directory + "/data__Dream-7B-Instruct")

    pattern = re.compile(
        r"^samples_.+_"
        r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)"
        r"\.jsonl$"
    )

    latest_time: datetime | None = None
    latest_paths: list[Path] = []

    for path in directory_path.glob("*.jsonl"):
        match = pattern.match(path.name)
        if not match:
            continue

        file_time = datetime.strptime(
            match.group(1),
            "%Y-%m-%dT%H-%M-%S.%f",
        )

        if latest_time is None or file_time > latest_time:
            latest_time = file_time
            latest_paths = [path]
        elif file_time == latest_time:
            latest_paths.append(path)

    return sorted(latest_paths)

def paired_bootstrap_test(base_correct, cadlm_correct, n_boot=1000, seed=42):
    base_correct = np.asarray(base_correct).astype(float)
    cadlm_correct = np.asarray(cadlm_correct).astype(float)
    assert base_correct.shape == cadlm_correct.shape

    rng = np.random.default_rng(seed)
    n = len(base_correct)

    observed_diff = cadlm_correct.mean() - base_correct.mean()

    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diff = cadlm_correct[idx].mean() - base_correct[idx].mean()
        boot_diffs.append(diff)

    boot_diffs = np.asarray(boot_diffs)
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    # two-sided bootstrap p-value around 0
    p_value = 2 * min(
        np.mean(boot_diffs <= 0),
        np.mean(boot_diffs >= 0)
    )
    p_value = min(p_value, 1.0)

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
    }

def load_results(file_path, dataset):
    data = load_jsonl(get_latest_json_paths(file_path))

    if dataset == 'humaneval_instruct':
        x = [(item['doc_id'], item['pass@1']) for item in data]
    elif dataset == 'mbpp_instruct':
        x = [(item['doc_id'], item['pass_at_1']) for item in data]
    elif dataset == 'gsm8k_cot':
        x = [(item['doc_id'], item['exact_match']) for item in data if item['filter'] == "flexible-extract"]
    elif dataset == 'minerva_math':
        x = [(item['doc_id'], item['math_verify']) for item in data]
    elif dataset == 'mmlu_pro':
        x = [(item['doc_id'], item['exact_match']) for item in data]
    elif dataset == 'gpqa_main_cot_zeroshot':
        x = [(item['doc_id'], item['exact_match']) for item in data if item['filter'] == "flexible-extract"]
    elif dataset == 'ifeval':
        x = [(item['doc_id'], sum([1 if a else 0 for a in item['inst_level_strict_acc']])) for item in data]
    
    x = sorted(x, key=lambda x: x[0])
    return [item[1] for item in x]


for target in [
    ('Base', 'attnAlign'),
    # ('Base', 'confidence'),
    # ('Dream-Lora', 'attnAlign'),
]:
    for dataset in [
        'humaneval_instruct', 'mbpp_instruct', 'gsm8k_cot', 'minerva_math', 
        'mmlu_pro', 'gpqa_main_cot_zeroshot', 'ifeval'
    ]:
        for k in [2, 4, 8]:
            target_file = f"evals_results_dream/{target[0]}/{dataset}_{k}_{target[1]}"
            src_file = f"evals_results_dream/Dream-Lora-Full_all_1e-02/{dataset}_{k}_attnAlign"

            base_correct = load_results(target_file, dataset)
            cadlm_correct = load_results(src_file, dataset)

            

            result = paired_bootstrap_test(base_correct, cadlm_correct)

            if result['p_value'] > 0.05:
                print(dataset, k, target, len(base_correct), len(cadlm_correct), result['p_value'], sum(base_correct), sum(cadlm_correct))
import os
import re
import json
import argparse
import prettytable as pt

from pathlib import Path
from datetime import datetime

def load_jsonl(path):
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as fr:
            return json.load(fr)
    data = []
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            data.append(json.loads(line))
    return data

def save_jsonl(data, path, mode='w'):
    with open(path, mode, encoding='utf-8') as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False) + '\n')

def get_latest_json_path(directory: str) -> str:

    directory = Path(directory)

    pattern = re.compile(
        r"^results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)\.json$"
    )

    latest_path = None
    latest_time = None

    for path in directory.glob("results_*.json"):
        match = pattern.match(path.name)
        if not match:
            continue

        time_str = match.group(1)
        dt = datetime.strptime(time_str, "%Y-%m-%dT%H-%M-%S.%f")

        if latest_time is None or dt > latest_time:
            latest_time = dt
            latest_path = path

    return latest_path

dataset_to_score_name = {
    'humaneval_instruct': 'pass@1,create_test',
    'mbpp_instruct': 'pass_at_1,extract_code',
    'gsm8k_cot': 'exact_match,flexible-extract',
    'minerva_math': 'math_verify,none',
    'truthfulqa_gen': 'rougeL_acc,none',
    'mmlu_pro': 'exact_match,custom-extract',
    'gpqa_main_cot_zeroshot': 'exact_match,flexible-extract',
    'ifeval': 'inst_level_strict_acc,none'
}

def load_score(path, name):
    results = [path.split('/')[-1]] + name.split('_')

    for dataset in [
        'humaneval_instruct', 'mbpp_instruct', 'gsm8k_cot', 'minerva_math', 
        'mmlu_pro', 'gpqa_main_cot_zeroshot', 'ifeval'
    ]:
        _dir = os.path.join(path, f"{dataset}_{name}")
        if not os.path.exists(_dir):
            results.append('N/A')
            continue
        
        _dir = os.path.join(_dir, os.listdir(_dir)[0])

        if not os.path.exists(_dir):
            results.append('N/A')
            continue

        file_path = get_latest_json_path(_dir)
        if file_path is not None:
            with open(file_path, 'r') as fr:
                data = json.load(fr)

                data = data['results'][dataset]
                results.append(round(data[dataset_to_score_name[dataset]] * 100, 1))
        else:
            results.append('N/A')
    
    return results

def score():
    scores = []

    if args.llada:
        base_name = 'evals_results_llada'
    elif args.illada:
        base_name = 'evals_results_illada'
    else:
        base_name = 'evals_results_dream'

    for lora_name in os.listdir(base_name):
        if args.filter is not None and lora_name not in args.filter:
            continue
        
        names = set()
        for _name in os.listdir(os.path.join(base_name, lora_name)):
            if args.method_filter is not None and '_'.join(_name.split('_')[-1:]) not in args.method_filter:
                continue
            names.add('_'.join(_name.split('_')[-2:]))
        names = list(names)
        
        for name in names:
            scores.append(load_score(os.path.join(base_name, lora_name), name))

    for score in scores:
        total_score = [s for s in score if isinstance(s, float)]
        mean_score = sum(total_score) / len(total_score) if total_score else 0.0
        score.append(round(mean_score, 1))
    
    scores.sort(key=lambda x: (
        x[1],
        x[-1]
    ))

    table = pt.PrettyTable()
    table.field_names = ['Model', 'NFE', 'Mode', 'HumanEval', 'MBPP', 'GSM8k', 'Math', 'MMLU-Pro', 'GPQA', 'IFEval', 'Average']
    table.align["Model"] = "l"

    for i, score in enumerate(scores):
        divider = i == len(scores) - 1 or \
            score[1] != scores[i + 1][1]
        table.add_row(score, divider=divider)
    
    print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--llada', action='store_true')
    parser.add_argument('-i', '--illada', action='store_true')

    parser.add_argument('-f', '--filter', nargs='+', default=None)

    parser.add_argument('-mf', '--method_filter', nargs='+', default=None)

    args = parser.parse_args()

    score()
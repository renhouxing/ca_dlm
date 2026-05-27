import os
import glob
import torch
import logging

from utils.utils import set_env, barrier
from utils.processor import Processor
from utils.trainer import DiffusionLMTrainer

from models import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from dataclasses import field, dataclass
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import HfArgumentParser, TrainingArguments

from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger()

@dataclass
class DiffusionTrainingArguments(TrainingArguments):

    # data
    max_len: int = field(default=2048)
    pad_len: int = field(default=8)
    num_workers: int = field(default=64)

    train_file: list[str] = field(default=None)

    # model
    model_cfg: str = field(default=None)
    lora: bool = field(default=False)
    
    # attn loss

    attn_alg: str = field(default="all")
    attn_coef: float = field(default=0)

    # t
    min_t: float = field(default=0.2)
    max_t: float = field(default=0.8)

    # train
    resume: bool = field(default=False)
    overwrite_output_dir: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        self.gradient_checkpointing_kwargs = {"use_reentrant": False}

        if self.attn_coef > 0:
            self.output_dir = f"{self.output_dir}_{self.attn_alg}_{self.attn_coef:.0e}"

def get_model(args):

    config = AutoConfig.from_pretrained(
        args.model_cfg,
        _attn_implementation="eager",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_cfg,
        config=config,
        dtype=torch.bfloat16, 
    )

    if args.lora:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            use_dora=True,
            lora_dropout=0.0,
            task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        )

        model = get_peft_model(model, peft_config)

        trainable_params, all_param = model.get_nb_trainable_parameters()

        logger.info(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )

    logger.info(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_cfg)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    processor = Processor(args.max_len, args.pad_len, tokenizer)

    return model, tokenizer, processor

def get_files(path):
    results = []
    for root, _, files in os.walk(path):
        for file in files:
            if '.jsonl' in file:
                results.append(os.path.join(root, file))
    return results

def tokenize_dataset(args, processor, paths):

    with args.main_process_first(desc="dataset map tokenization", local=False):

        train_sets = []

        for path in paths:

            if os.path.exists(os.path.join(path, 'dataset_info.json')):
                dataset = load_from_disk(path)
                logger.info('Load dataset from disk %s', path)
            else:
                if os.path.isdir(path):
                    src = get_files(path)
                else:
                    src = glob.glob(path)
                
                logger.info("Loading dataset from files: %s", src)
                dataset = load_dataset('json', data_files=src, split='train')
                dataset = dataset.map(
                    processor.process_tokenize,
                    batched=True,
                    batch_size=8192,
                    num_proc=args.num_workers,
                    remove_columns=list(dataset.features),
                    desc="Running tokenizer on dataset",
                )
            
            logger.info("Dataset size: %d", len(dataset))
            train_sets.append(dataset)
    
        train_sets = concatenate_datasets(train_sets)
    
    logger.info("Train dataset size: %d", len(train_sets))

    return train_sets

def train():
    parser = HfArgumentParser(DiffusionTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    set_env(args)

    model, tokenizer, processor = get_model(args)

    train_set = tokenize_dataset(args, processor, args.train_file)

    if args.do_eval:
        split_set = train_set.train_test_split(test_size=0.05, seed=args.seed)
        dataset = dict(train_dataset=split_set['train'], eval_dataset=split_set['test'])
    else:
        dataset = dict(train_dataset=train_set)

    trainer = DiffusionLMTrainer(
        args=args,
        model=model, 
        processing_class=tokenizer,
        **dataset
    )

    has_checkpoint = len(glob.glob(os.path.join(args.output_dir, "checkpoint-*"))) > 0

    trainer.train(resume_from_checkpoint=has_checkpoint)
    trainer.save_model(os.path.join(args.output_dir, "checkpoint-final"))

    barrier()

if __name__ == "__main__":

    try:
        train()
    except Exception as e:
        logging.exception(e)

        exit(-1)

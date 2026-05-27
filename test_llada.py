
import os
import math
import time
import torch
import torch.nn.functional as F

from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate

from models import AutoTokenizer, AutoConfig, AutoModelForCausalLM

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

@register_model("attn_dlm")
class AttnMaskLM(LM):

    def __init__(
        self,
        pretrained="data/Dream-7B-Instruct",
        lora_path=None,
        generation_len=512,
        pred_per_step=4,
        score_mode="attn",
        temperature=0.0,
        top_p=0.9,
        device="cuda",
        dtype="bfloat16",
        **kwargs,
    ):
        super().__init__()

        self.pretrained = pretrained
        self.generation_len = int(generation_len)
        self.pred_per_step = int(pred_per_step)
        self.score_mode = score_mode
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        accelerator_kwargs = InitProcessGroupKwargs()
        self.accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])

        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes

        self._device = torch.device(f"cuda:{self._rank}")

        if dtype in ["bf16", "bfloat16"]:
            torch_dtype = torch.bfloat16
        elif dtype in ["fp16", "float16"]:
            torch_dtype = torch.float16
        elif dtype in ["fp32", "float32"]:
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        config = AutoConfig.from_pretrained(
            pretrained,
            _attn_implementation="eager",
        )
        config.attn_layer = 1000

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            config=config,
            dtype=torch_dtype,
        ).to(self._device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

        if lora_path not in [None, "", "None", "none"]:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                device_map="auto",
            )
            self.model = self.model.merge_and_unload()
            self.model.eval()

        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.mask_id = self.tokenizer.mask_token_id

        if self.mask_id is None:
            raise ValueError("tokenizer.mask_token_id is None; mask diffusion decoding needs a mask token.")
    
    def all_gather(self, tensor):
        if self.world_size <= 1:
            return tensor
        return self.accelerator.gather(tensor)

    def gather_object(self, obj, dst=0):
        if self.world_size <= 1:
            return [obj]
        result = [None] * self.world_size if self.rank == dst else None
        torch.distributed.gather_object(obj=obj, object_gather_list=result, dst=dst)
        return result

    def barrier(self):
        if self.world_size > 1:
            self.accelerator.wait_for_everyone()

    @property
    def device(self):
        return self._device

    @property
    def tokenizer_name(self):
        return getattr(self.tokenizer, "name_or_path", self.pretrained)

    def chat_template(self, chat_template=False):
        return self.tokenizer.chat_template

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @torch.no_grad()
    def generate_until(self, requests):
        """
        For generate_until tasks:
          GSM8K generate
          HumanEval generate
          MBPP generate
          many CoT/math/code generation YAMLs

        Each request.args is:
          (context: str, gen_kwargs: dict)
        """
        outputs = []

        start_time, total = time.time(), len(requests)
        for i, req in enumerate(requests, start=1):
            context, gen_kwargs = req.args

            if 'Dream' in self.pretrained:
                if req.task_name in ['humaneval_instruct', 'mbpp_instruct']:
                    context = context[:-len("<|im_end|>\n")]
                    print(context.encode())
            else:
                if req.task_name in ['humaneval_instruct', 'mbpp_instruct']:
                    context = context[:-len("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")]

            input_ids = self.tokenizer(
                context,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0].to(self.device)

            input_len = input_ids.shape[0]

            output_ids = self.mask_diffusion(
                input_ids=input_ids,
                max_gen_len=self.generation_len,
                pred_per_step=self.pred_per_step,
                score_mode=self.score_mode,
            )

            text = self.tokenizer.decode(
                output_ids[input_len:],
                skip_special_tokens=False,
            )

            text = (
                text.replace("<|endoftext|>", "")
                    .replace("<|mask|>", "")
                    .replace("<|mdm_mask|>", "")
            )

            if req.task_name == 'mbpp_instruct':
                text = text.split('```')[0]

            now_time = time.time()
            rest_time = (now_time - start_time) / i * (total - i) / 60
            print(f"Rank {self._rank} Processing {i}/{total}, Rest Time: {rest_time:.2f} mins")

            outputs.append(text)
            self.cache_hook.add_partial("generate_until", req.args, text)

        return outputs

    @torch.no_grad()
    def loglikelihood(self, requests):
        """
        Needed by multiple-choice / cloze tasks:
          mmlu, hellaswag, arc_easy, arc_challenge, winogrande, piqa, etc.

        Important:
        This is a teacher-forcing pseudo-loglikelihood implementation.
        For a true diffusion LM, this may not be theoretically identical to AR likelihood.
        But it lets lm_eval built-in multiple_choice YAMLs run.
        """
        results = []

        for req in tqdm(requests, disable=(self.rank != 0), desc="loglikelihood"):
            context, continuation = req.args

            if context == "":
                context_enc = []
            else:
                context_enc = self.tokenizer.encode(
                    context,
                    add_special_tokens=False,
                )

            continuation_enc = self.tokenizer.encode(
                continuation,
                add_special_tokens=False,
            )

            if len(continuation_enc) == 0:
                results.append((0.0, True))
                continue

            input_ids = torch.tensor(
                context_enc + continuation_enc,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            outputs = self.model(
                input_ids=input_ids,
                is_causal=False,
                use_cache=False,
            )

            logits = outputs.logits[0]

            ctx_len = len(context_enc)
            cont_len = len(continuation_enc)

            logprob_sum = 0.0
            greedy = True

            for i in range(cont_len):
                token_pos = ctx_len + i
                token_id = continuation_enc[i]

                if token_pos == 0:
                    score_pos = 0
                else:
                    score_pos = token_pos - 1

                token_logits = logits[score_pos]
                log_probs = F.log_softmax(token_logits.float(), dim=-1)

                logprob_sum += log_probs[token_id].item()

                pred_id = torch.argmax(token_logits).item()
                if pred_id != token_id:
                    greedy = False

            results.append((logprob_sum, greedy))
            self.cache_hook.add_partial("loglikelihood", req.args, (logprob_sum, greedy))

        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests):
        """
        Needed by perplexity-style tasks.
        If you don't run perplexity tasks, this basically won't matter.
        """
        results = []

        for req in tqdm(requests, disable=(self.rank != 0), desc="rolling loglikelihood"):
            (text,) = req.args

            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )

            if len(token_ids) <= 1:
                results.append(0.0)
                continue

            input_ids = torch.tensor(
                token_ids,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            outputs = self.model(
                input_ids=input_ids,
                is_causal=False,
                use_cache=False,
            )

            logits = outputs.logits[0]

            logprob_sum = 0.0
            for pos in range(1, len(token_ids)):
                token_id = token_ids[pos]
                token_logits = logits[pos - 1]
                log_probs = F.log_softmax(token_logits.float(), dim=-1)
                logprob_sum += log_probs[token_id].item()

            results.append(logprob_sum)
            self.cache_hook.add_partial("loglikelihood_rolling", req.args, logprob_sum)

        return results

    def sample(self, logits, temperature=0.0, top_p=1.0):
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D [B, V], got shape {logits.shape}")

        if temperature < 0:
            raise ValueError("temperature must be >= 0")

        if not (0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")

        if temperature == 0:
            return torch.argmax(logits, dim=-1), F.softmax(logits, dim=-1)

        logits = logits / temperature

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(
                dim=-1,
                index=sorted_indices,
                src=sorted_indices_to_remove,
            )

            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_token_id, probs

    def select_tokens(self, weights, corr, m):
        if m == 1:
            return torch.argmax(weights).view(1)

        selected = []
        for _ in range(m):
            coefs = weights.clone()

            if selected:
                coefs[selected] = -torch.inf
                coefs = coefs - corr[:, selected].sum(dim=1)

            k = torch.argmax(coefs).item()
            selected.append(k)

        return torch.tensor(selected, dtype=torch.long, device=weights.device)

    def sample_tokens(
        self,
        logits,
        attn_weights,
        score_mode,
        answer_mask,
        pred_this_step,
    ):
        attn_weights = torch.cat(attn_weights, dim=0).mean(dim=0).mean(dim=0)

        curr_token_ids, probs = self.sample(
            logits,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        if score_mode == "confidence":
            scores = torch.gather(probs, dim=-1, index=curr_token_ids.unsqueeze(-1)).squeeze(-1)
            scores[~answer_mask] = -torch.inf
            scores[curr_token_ids == self.pad_id] = -torch.inf
            pred_positions = torch.topk(scores, k=pred_this_step).indices

        elif score_mode == "entropy":
            scores = torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            scores[~answer_mask] = -torch.inf
            pred_positions = torch.topk(scores, k=pred_this_step).indices

        elif score_mode == "uniform":
            scores = torch.rand_like(curr_token_ids, dtype=torch.float)
            scores[~answer_mask] = -torch.inf
            scores[curr_token_ids == self.pad_id] = -torch.inf
            pred_positions = torch.topk(scores, k=pred_this_step).indices

        elif score_mode == "dos":
            scores = attn_weights[:, ~answer_mask].mean(dim=1)
            scores[~answer_mask] = -torch.inf
            pred_positions = torch.topk(scores, k=pred_this_step).indices

        elif score_mode == "attnAlign":
            weights = attn_weights[answer_mask][:, ~answer_mask].sum(dim=1)
            corr = attn_weights[answer_mask, :][:, answer_mask]

            selected = self.select_tokens(weights, corr, m=pred_this_step)
            pred_positions = torch.where(answer_mask)[0][selected]
        
        elif score_mode == "attnAlignLogit":

            weights = attn_weights[answer_mask][:, ~answer_mask].sum(dim=1)
            corr = attn_weights[answer_mask, :][:, answer_mask]

            scores = torch.gather(probs, dim=-1, index=curr_token_ids.unsqueeze(-1)).squeeze(-1)
            scores[curr_token_ids == self.pad_id] /= 2
            scores = scores[answer_mask]
            
            weights = weights * scores

            selected = self.select_tokens(weights, corr, m=pred_this_step)
            pred_positions = torch.where(answer_mask)[0][selected]

        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")

        return pred_positions, curr_token_ids

    def mask_diffusion(
        self,
        input_ids,
        max_gen_len=256,
        pred_per_step=4,
        score_mode="attn",
    ):
        input_len = input_ids.shape[0]

        if input_len > 2048:
            print(input_len)
            return input_ids

        mask_token = torch.full(
            (max_gen_len,),
            self.mask_id,
            dtype=torch.long,
            device=input_ids.device,
        )

        input_ids = torch.cat([input_ids, mask_token], dim=-1)
        answer_mask = input_ids == self.mask_id

        steps = math.ceil(max_gen_len / pred_per_step)

        for i in range(steps):
            outputs = self.model(
                input_ids=input_ids.unsqueeze(0),
                is_causal=False,
                use_cache=False,
                output_attentions=True,
            )

            pred_positions, curr_token_ids = self.sample_tokens(
                outputs.logits[0],
                outputs.attentions,
                score_mode,
                answer_mask,
                pred_per_step,
            )

            answer_mask[pred_positions] = False
            input_ids[pred_positions] = curr_token_ids[pred_positions]

            if self.eos_id is not None and self.eos_id in input_ids[input_len:]:
                eos_positions = (input_ids[input_len:] == self.eos_id).nonzero(as_tuple=True)[0]
                eos_index = eos_positions[0].item() + input_len

                if (input_ids[:eos_index] != self.mask_id).all():
                    break

        return input_ids

if __name__ == "__main__":
    cli_evaluate()
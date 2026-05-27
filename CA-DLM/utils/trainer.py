import torch
import logging
import datetime

import torch.distributed as dist
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

from transformers import Trainer, TrainerCallback

logger = logging.getLogger()

class LoggerCallback(TrainerCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        
        self.start_time = datetime.datetime.now()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero:
            return
        
        loss_msg = ' '.join(["%s: %.4f" % (k, v) for k, v in logs.items() if 'loss' in k or 't' == k])

        if loss_msg == '':
            return
        
        now = datetime.datetime.now()
        pass_time = now - self.start_time
        rest_time = pass_time * (state.max_steps - state.global_step) / max(1, state.global_step)
        eta = now + rest_time

        pt_min = pass_time.seconds // 60
        pass_time = '%.2d:%.2d' % (pt_min // 60 + pass_time.days * 24, pt_min % 60)

        rt_min = rest_time.seconds // 60
        rest_time = '%.2d:%.2d' % (rt_min // 60 + rest_time.days * 24, rt_min % 60)

        logger.info(
            'step: %d epoch: %.4f %s lr: %.4g passed time: %s rest time: %s eta: %s',
            state.global_step, state.epoch, loss_msg, logs.get('learning_rate', 0),
            pass_time, rest_time, eta.strftime('%m/%d %H:%M')
        )

class Collator:

    def __init__(self, args, tokenizer):

        self.args = args
        self.tokenizer = tokenizer

        self._mask_id = tokenizer.mask_token_id

    def __call__(self, inputs):
        input_ids, labels, attention_mask, prompt_mask = [], [], [], []

        t = torch.rand(1).item() * (self.args.max_t - self.args.min_t) + self.args.min_t

        for _input in inputs:
            _input_ids = torch.tensor(_input['input_ids']).long()
            _labels = _input_ids.clone()
            
            mask = torch.rand_like(_input_ids, dtype=torch.float) < t

            mask[:_input['prompt_lens']] = False

            _input_ids[mask] = self._mask_id
            _labels[~mask] = -100

            input_ids.append(_input_ids)
            labels.append(_labels)
            attention_mask.append(torch.ones_like(input_ids[-1]))
            prompt_mask.append(torch.arange(len(input_ids[-1])) < _input['prompt_lens'])
        
        return {
            "t": t,
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "target": pad_sequence(labels, batch_first=True, padding_value=-100),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "prompt_mask": pad_sequence(prompt_mask, batch_first=True, padding_value=False),
        }

class DiffusionLMTrainer(Trainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_callback(LoggerCallback)
        self.data_collator = Collator(self.args, self.processing_class)

        self._stored_metrics = {}

        self.torch_generator = torch.Generator()
        self.torch_generator.manual_seed(self.args.seed)

        self.mask_id = self.processing_class.mask_token_id
    
    def reduce_tensor(self, tensor):

        if not dist.is_initialized():
            return tensor.detach().nanmean().item()

        world_size = dist.get_world_size()

        if world_size <= 1:
            return tensor.detach().nanmean().item()

        tensor = tensor.detach().nanmean()
        tensors = [torch.empty_like(tensor) for _ in range(world_size)]

        dist.all_gather(tensors, tensor)
        tensor = torch.stack(tensors, dim=0).nanmean()

        return tensor.item()
    
    def store_metrics(self, metrics):
        for key, value in metrics.items():
            if key not in self._stored_metrics:
                self._stored_metrics[key] = []
            self._stored_metrics[key].append(value)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['target'],
            use_cache=False, 
            is_causal=False,
            output_attentions=True,
            num_items_in_batch=num_items_in_batch
        )

        loss = outputs.loss.clone() 

        if self.args.attn_coef > 0:
            masked = inputs["target"] != -100                  # [B, S]
            valid = inputs["attention_mask"].bool()            # [B, S]
            visible = (~masked) & valid                        # [B, S]

            # [B, L, H, S, S] -> [B, S, S]
            attn = torch.stack(outputs.attentions, dim=1).mean(dim=1).mean(dim=1)

            B, S, _ = attn.shape

            # remove self-attention for dependency-style loss
            eye = torch.eye(S, device=attn.device, dtype=torch.bool).unsqueeze(0)
            attn = attn.masked_fill(eye, 0)

            pred = outputs.logits.argmax(dim=-1)               # [B, S]
            correct = (pred == inputs["target"]) & masked      # [B, S]
            wrong = (pred != inputs["target"]) & masked        # [B, S]

            # score for every query token

            score_visible = attn.masked_fill(~visible.unsqueeze(1), 0).sum(dim=-1)
            score_masked = attn.masked_fill(~masked.unsqueeze(1), 0).sum(dim=-1)

            # correct masked query: attend more to visible context than masked region
            correct_margin = score_visible - score_masked
            right_loss_vec = -F.logsigmoid(correct_margin)

            # wrong masked query: attend more to correct masked tokens than wrong masked tokens

            score_mask_right = attn.masked_fill(~correct.unsqueeze(1), 0).sum(dim=-1)
            score_mask_wrong = attn.masked_fill(~wrong.unsqueeze(1), 0).sum(dim=-1)

            wrong_margin = score_mask_right - score_mask_wrong
            wrong_loss_vec = -F.logsigmoid(wrong_margin)

            if self.args.attn_alg == "right":
                attn_loss = right_loss_vec[correct].mean()
            elif self.args.attn_alg == "wrong":
                attn_loss = wrong_loss_vec[wrong].mean()
            elif self.args.attn_alg == "all":
                right_loss = right_loss_vec[correct].sum()
                wrong_loss = wrong_loss_vec[wrong].sum()
                denom = masked.sum().clamp_min(1)
                attn_loss = (right_loss + wrong_loss) / denom

            loss += self.args.attn_coef * attn_loss

        self.store_metrics({
            "loss": self.reduce_tensor(loss),
        })

        if self.args.attn_coef > 0:
            self.store_metrics({
                "lm_loss": self.reduce_tensor(outputs.loss),
                "attn_loss": self.reduce_tensor(attn_loss),
            })

        return (loss, {}) if return_outputs else loss
    
    def log(self, logs, start_time=None):
        logs.pop('loss', None)
        for key, metrics in self._stored_metrics.items():
            if len(metrics) > 0:
                logs[key] = torch.tensor(metrics).mean().item()
        
        for key in self._stored_metrics:
            self._stored_metrics[key].clear()
        
        super().log(logs, start_time=None)
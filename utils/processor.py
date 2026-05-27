class Processor:

    def __init__(self, max_len, pad_len, tokenizer):

        self.max_len = max_len
        self.pad_len = pad_len
        self.tokenizer = tokenizer

    def process_tokenize(self, examples):

        input_ids, prompt_lens = [], []
        for messages in examples['messages']:
            inputs_id = self.tokenizer.apply_chat_template(messages)
            prompt_id = self.tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)

            resp_id_len = len(inputs_id) - len(prompt_id)
            if resp_id_len % self.pad_len != 0:
                pad_num = self.pad_len - resp_id_len % self.pad_len
                inputs_id += [self.tokenizer.pad_token_id] * pad_num
            
            if len(inputs_id) <= self.max_len and resp_id_len >= 32:
                input_ids.append(inputs_id)
                prompt_lens.append(len(prompt_id))
        
        return {
            "input_ids": input_ids,
            "prompt_lens": prompt_lens,
        }
    
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto")

    def generate(self, prompt):
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
            out = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return out
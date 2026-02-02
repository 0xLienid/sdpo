from transformers import AutoModelForCausalLM, AutoTokenizer


class Validator:
    def __init__(self, name: str):
        self.name = name

    def validate(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, batch_size: int, max_new_tokens: int):
        pass

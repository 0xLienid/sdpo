import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from validators.livecodebench.livecodebench_validator import LiveCodeBenchValidator


def test_livecodebench_validator():
    batch_size = 1
    max_new_tokens = 1024

    validator = LiveCodeBenchValidator()
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    result = validator.validate(model, tokenizer, batch_size, max_new_tokens)
    assert result > 0.0
    print(f"LiveCodeBenchValidator result: {result}")


if __name__ == "__main__":
    test_livecodebench_validator()

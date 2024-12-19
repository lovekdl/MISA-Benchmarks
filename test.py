from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
)
import torch
model_name = "meta-llama/Meta-Llama-3-8B"
config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        device_map='auto',
        config=config,
        torch_dtype=torch.float16,
    )

data = load_dataset('datasets/mawps')
print(data)
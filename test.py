from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
)
import torch
model_name = "meta-llama/Meta-Llama-3-8B"
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        config=config,
        torch_dtype=torch.float16,
    )
sum = 0
for n, p in model.named_parameters() :
    sum += p.numel()
    print(n,p.shape,p.numel())
print(sum)
# data = load_dataset('datasets/mawps')
# print(data)
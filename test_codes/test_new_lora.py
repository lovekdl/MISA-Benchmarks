from datasets import load_dataset

# from utils.lora_module import convert_linear_layer_to_lora,only_optimize_lora_parameters
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
model_name = "Qwen/Qwen2.5-7B-Instruct"
# model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "mistralai/Mistral-7B-v0.3"
model_name = "Qwen/Qwen2.5-7B"
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        config=config,
        torch_dtype=torch.float16,
    )
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)

from utils.lora_module import convert_linear_layer_to_lora,only_optimize_lora_parameters
model = convert_linear_layer_to_lora(model, ['q_proj', 'up_proj', 'down_proj'], 16, lora_scaling = 32, lora_dropout=0.05)
model = only_optimize_lora_parameters(model)
print(model.config)
# exit(0)
sum = 0
for n, p in model.named_parameters() :
    sum += p.numel()
print(sum)
for n, p in model.named_parameters() :
    print(n,p.shape, len(p.shape), p.numel()/sum, p.dtype)
# data = load_dataset('datasets/mawps')
# print(data)
exit(0)

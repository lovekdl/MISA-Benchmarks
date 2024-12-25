from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
)
import torch
model_name = "meta-llama/Llama-2-7b-hf"
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
print(sum)
for n, p in model.named_parameters() :
    print(n,p.shape, p.numel()/sum)
# data = load_dataset('datasets/mawps')
# print(data)

peft_kwargs = {
    "r": 16,
    "target_modules": ['q_proj','k_proj','v_proj','o_proj','up_proj','down_proj', 'gate_proj'],
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model
# lora_config = LoraConfig(
        # task_type=TaskType.SEQ_CLS,
#         inference_mode=False,
#         use_dora=False,
#         **peft_kwargs,
#     )
# model = get_peft_model(model, lora_config)

sum = 0
lora_sum = 0
layer_sum = 0
for n, p in model.named_parameters() :
    sum += p.numel()
    if("lora" in n) :
        lora_sum += p.numel()
    if '31' in n :
        layer_sum += p.numel()
print(sum)
for n, p in model.named_parameters() :
    print(n,p.shape, p.numel()/sum)

print(f"Lora Learnable: {lora_sum/sum}")
print(f"layer Learnable: {layer_sum / sum }")
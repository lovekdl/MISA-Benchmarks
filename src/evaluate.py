from src.optimizer import BlockCoordinateDesentOptimizer
from torch.optim import AdamW
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
)
model_name = "meta-llama/Meta-Llama-3-8B"
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        config=config,
        torch_dtype=torch.float16,
    )
base_optimizer = AdamW(model.parameters(), lr=5e-6)
optimizer = BlockCoordinateDesentOptimizer(
                base_optimizer=base_optimizer,
                named_parameters_list=list(model.named_parameters()))


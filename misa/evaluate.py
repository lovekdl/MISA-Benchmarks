from typing import List

import torch
import transformers
from datasets import load_dataset
import wandb
from dataclasses import dataclass, field
from misa.custom_trainer import CustomizedTrainer
from misa.args import CustomizedTrainingArguments
from misa.callbacks import MemoryTimeConsumptionCallback, MathReasoningEvaluateCallback, CommonsenseReasoningEvaluateCallback
from peft import (  # noqa: E402
    LoraConfig, 
    TaskType, 
    get_peft_model,
)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    # MultiLingAdapterArguments,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from misa.args import CustomizedTrainingArguments
import datetime
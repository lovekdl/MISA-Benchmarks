from typing import List, Optional, Union
import wandb
from dataclasses import dataclass, field
from transformers import TrainingArguments
@dataclass
class CustomizedTrainingArguments(TrainingArguments) :
    # optimizer : str = 'adamw'
    optimizer: Optional[str] = field(
        default='adamw',
        metadata = {
            "help": ""
        },
    )
    grad_importance_exp: Optional[float] = field(
        default=1,
        metadata = {
            "help": ""
        },
    )

    bcd_activated_layers: Optional[int] = field(
        default=1,
        metadata = {
            "help": ""
        },
    )

    bcd_interval_steps: Optional[int] = field(
        default=30,
        metadata = {
            "help": ""
        },
    )
    bcd_update_order: Optional[str] = field(
        default='ascending',
        metadata = {
            "help": ""
        },
    )
    bcd_target_attn: bool = field(
        default=True,
        metadata={"help": ""},
    )
    bcd_target_mlp: bool = field(
        default=True,
        metadata={"help": ""},
    )
    bcd_base_optimizer:str = field(
        default='adamw',
        metadata={"help": ""},
    )
    granularity:str = field(
        default='layer',
        metadata={"help": ""},
    )
    only_layer:Optional[int] = field(
        default=-1,
        metadata={"help": ""},
    )
    offload_optimizer_state:Optional[bool] = field(
        default=False,
        metadata={"help": ""}
    )
    bcd_suffix_start_index:Optional[int] = field(
        default=0,
        metadata={"help": ""} ,
    )
    offload_rank:Optional[int]=field(
        default=-1,
        metadata={"help":""},
    )
    offload_quantization_bit:Optional[int]=field(
        default=-1,
        metadata={"help":""},
    )
    param_ratio_limit: Optional[float] = field(
        default=1,
        metadata={"help":""}
    )
    LRU: Optional[int] = field(
        default=0,
        metadata={"help":""}
    )
    normalization_type: str = field(
        default="L-norm",
        metadata={"help":""}
    )
    module_target: str = field(
        default="all",
        metadata={"help":""}
    )
    testing_memory: str = field(
        default='n',
        metadata={"help":"y or n"}
    )
    every_epoch_eval: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    mix_lora:bool=field(
        default=False,
        metadata={
            "help": ""
        },
    )

    # Galore
    galore:bool = field(
        default=False,
        metadata={"help": "whether to use galore"},
    )
    galore_r: Optional[int] = field(
        default=16,
        metadata={"help": "rank of galore"},
    )
    galore_alpha: Optional[float] = field(
        default=1.0,
        metadata={"help": "alpha of galore"},
    )
    include_embedding_and_lm_head: Optional[bool] = field(
        default=False,
        metadata= {"help": "lisa or not"}
    )

import torch
from torch.optim import SGD, Adam, AdamW

from transformers import Trainer
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)


from transformers.utils import (
    logging,
)
from src.optimizer import BlockCoordinateOptimizer
from src.clip_grad_norm import clip_grad_norm_for_sparse_tensor
from types import MethodType

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

logger = logging.get_logger(__name__)

class OurTrainer(Trainer):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.optimizer == 'bcd-optimizer':
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)
    
    def create_optimizer(self) -> "torch.optim.Optimizer":
        args = self.args
        print(f"Optimizer is {args.optimizer}")
        if args.optimizer == "bcd-optimizer" : 
            # if args.optimizer == "adam" :
            #     base_optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
            # elif args.optimizer == "sgd":
            #     base_optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
            # elif args.optimizer == "adamw":
            #     base_optimizer = AdamW(self.model.parameters(), lr=args.learning_rate)
            # else : raise NotImplementedError("We only support 3 optimizers: adam, sgd and adamw. ")
            
            base_optimizer = AdamW(self.model.parameters(), lr=args.learning_rate)
            # print(BlockCoordinateOptimizer.__module__)
            # print(args.offload_optimizer_state)
            self.optimizer = BlockCoordinateOptimizer(
                base_optimizer=base_optimizer,
                named_parameters_list=list(self.model.named_parameters()),
                bcd_activated_layers=args.bcd_activated_layers,
                bcd_interval_steps=args.bcd_interval_steps,
                bcd_order=args.bcd_update_order,
                block_target_attn=args.bcd_target_attn,
                block_target_mlp=args.bcd_target_mlp,
                block_target_non_linear=False,
                only_layer=args.only_layer,
                offload_optimizer_state=args.offload_optimizer_state,
                grad_importance_exp = args.grad_importance_exp,
                bcd_suffix_start_index = args.bcd_suffix_start_index,
                device=self.model.device,
                offload_rank=args.offload_rank,
                offload_quantization_bit=args.offload_quantization_bit,
                granularity=args.granularity,
            )
        elif args.optimizer == "adam":
                self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
        elif args.optimizer == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
        elif args.optimizer == "adamw":
            self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate)
        else : raise NotImplementedError("We only support 3 optimizers: adam, sgd and adamw. ")
                
        return super().create_optimizer()

    
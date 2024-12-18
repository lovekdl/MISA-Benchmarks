import torch
import random
from torch.optim import Optimizer
from torch import Tensor
from collections import defaultdict
from typing import List, Optional, Dict, Union, Iterable
import time
import math
import warnings
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import numpy as np
import wandb

BACKWARD_VERBOSE = 0
from torch.optim import AdamW
class BlockCoordinateOptimizer(Optimizer):
    """Wrap the original optimizer to update trainable parameters periodically based on number of activated layers."""

    def __init__(
        self,
        base_optimizer: Optimizer,
        named_parameters_list,
        bcd_interval_steps = 5,
        active_modules: List[str] = [],
        include_embedding_and_lm_head=False,
        bcd_activated_layers=1,
        block_target_attn=True,
        block_target_mlp=True,
        block_target_non_linear=True,
        offload_optimizer_state=False,
        only_layer=-1,
        bcd_order="ascending",
        verbose: int = 1,
        grad_beta = 0.9,
        grad_importance_exp = 1, 
        bcd_suffix_start_index = 0,
        device = 'cuda',
        offload_rank = -1,
        offload_quantization_bit = 8, 
        log_fn = None,
    ):
        """
        Args:
            base_optimizer (Optimizer): The base optimizer being wrapped by the BlockOptimizer.
            named_parameters_list: A function that generates the named parameters of the model.
            block_prefix_list (List[List[str]]): The list of blocks of parameters to be updated.
            bcd_interval_steps (int, optional): The number of optimization steps before switching to the next block. Defaults to 10.
            start_block (Optional[int], optional): The index of the block to start with. Defaults to None.
            active_modules (List[str]): The list of modules that are always active during optimization. Defaults to None.
            verbose (int, optional): The verbosity level for printing information during optimization. Defaults to 1.
            log_fn: A logging function for recording information during optimization. Defaults to None.
        """
        # print(base_optimizer)
        block_prefix_list, other_params = self.infer_param_groups([n for n, _ in named_parameters_list])
        
        assert isinstance(block_prefix_list, list)
        self.bcd_activated_layers = bcd_activated_layers
        self.bcd_interval_steps = bcd_interval_steps
        self.verbose = verbose
        self.named_parameters_list = named_parameters_list
        self.weight_decay = base_optimizer.param_groups[0]["weight_decay"]
        self.block_prefix_list = block_prefix_list
        self.other_params = other_params
        self.log_fn = log_fn
        self.global_step = 0
        self.base_optimizer = base_optimizer
        self.active_modules = active_modules
        self.defaults = base_optimizer.defaults
        self.active_layers_indices = []
        self.include_embedding_and_lm_head = include_embedding_and_lm_head
        self.only_layer = only_layer
        self.bcd_order = bcd_order

        self.device=device
        self.offload_rank = offload_rank
        # train a suffix layers of the model
        self.bcd_suffix_start_index = bcd_suffix_start_index
        self.total_layers = self.total_layers - bcd_suffix_start_index
        self.block_prefix_list=self.block_prefix_list[bcd_suffix_start_index:]

        self.offload_quantization_bit = offload_quantization_bit
        self.current_grad_norms = [[] for _ in range(self.total_layers)]
        self.grad_norms = [0] * self.total_layers
        self.last_grad_norms = [100000.0] * self.total_layers
        self.avg_grad_norms = [0] * self.total_layers
        self.avg_grad_norms_hat = [0] * self.total_layers
        self.grad_norms_calculated_times = [0] * self.total_layers
        self.embed_grad_norm = 0
        self.avg_embed_grad_norm_hat = 0
        self.embed_grad_norm_calculated_times = 0
        self.lm_head_grad_norm = 0
        self.lm_head_grad_norm_calculated_times = 0
        self.avg_lm_head_grad_norm_hat = 0

        self.lp_to_hp = {}

        self.layers_param_number = [0] * self.total_layers
        self.embed_param_number = 0
        self.lm_head_param_number = 0
        self.calculate_param_numbers(named_parameters_list)

        self.layer_selected_times = [0] * self.total_layers

        self.param_groups = base_optimizer.param_groups
        self.state_dict = base_optimizer.state_dict # for compatibility of hf Trainer
    
        self.permutation = []
        self.block_target_attn=block_target_attn
        self.block_target_mlp=block_target_mlp
        self.block_target_non_linear=block_target_non_linear
        self.offload_optimizer_state = offload_optimizer_state

        self.grad_beta = grad_beta
        self.layers_grad_beta_multiple = [grad_beta] * self.total_layers
        self.embed_grad_beta_multiple = grad_beta
        self.lm_head_grad_beta_multiple = grad_beta
        # lora not supported in here
        self.lora_mode = False

        self.grad_importance_exp = grad_importance_exp
        
        self.param_to_optimizer_state = {}
        # fp32_params = []
        # for n, p in named_parameters_list:
        #     if p.dtype == torch.float32:
        #         fp32_params.append(n)

        if any(isinstance(p, torch.FloatTensor) for _, p in named_parameters_list):
            warnings.warn("Expect model to be loaded in fp16 precision while detect fp32 weight. \
                This will cause additional memory usage and lose the benefit of mixed precision training.")
            
        super().__init__(self.param_groups, base_optimizer.defaults)
        
        if BACKWARD_VERBOSE:
            self.record_mark = True
            self.ordered_named_params = []
            self.param_num = len(named_parameters_list)
            for n, p in named_parameters_list:
                p.register_post_accumulate_grad_hook(self.test_hook(n))

        self.update_trainable_params()

        if BACKWARD_VERBOSE == 2:
            for name, param in self.named_parameters_list:
                param.requires_grad_(True)
    
    @property
    def embedding_layer(self):
        for n, p in self.named_parameters_list:
            if "embed" in n:
                return p
    
    @property
    def lm_head_layer(self):
        for n, p in self.named_parameters_list:
            if "lm_head" in n:
                return p

    def infer_param_groups(self, param_names):
        """automatic inference of the parameter groups based on the parameter names.
        divide groups into:
            * embedding
            * transformer layers
            * lm_head and others
        """
        import re
        
        block_prefix_list = []
        other_params = []
        layers_pattern = r'.*layers.[^.]*\.'
        layer_pattern = r'.*layer.[^.]*\.'

        for name in param_names:
            # print(name, block_prefix_list)
            if any(prefix[0] in name for prefix in block_prefix_list):
                continue
            
            if re.findall(layers_pattern, name) and "lm_head" not in name:
                block_prefix_list.append(re.findall(layers_pattern, name))
            elif re.findall(layer_pattern, name) and "lm_head" not in name:
                block_prefix_list.append(re.findall(layer_pattern, name))
            else: 
                other_params.append(name)
            # elif re.findall(embed_pattern, name) and include_embedding:
            #     embed_list.append(re.findall(embed_pattern, name)[0])
            # else:
            #     lm_head_and_other_params.append(name)
        
        # if include_lm_head:
        #     block_prefix_list.append(lm_head_and_other_params)
        # print("checking")
        # for i in range(10) :
        #     print(block_prefix_list)
        # print(block_prefix_list)
        self.total_layers = len(block_prefix_list)
        # print(other_params)
        return block_prefix_list, other_params
    


    def calculate_param_numbers(self, named_param_list) :
        embed_pattern = r'.*embed[^.]*\.'
        layer_pattern = r'.*layers.[^.]*\.'
        import re

        self.current_grad_norms = [[] for _ in range(self.total_layers)]
        for name, param in self.named_parameters_list:
            is_layer_param = False
            for i in range(self.total_layers) :
                if(self.block_prefix_list[i][0] in name) :
                    self.layers_param_number[i] += param.numel()
                    is_layer_param = True
                    break
            if not is_layer_param :
                if re.findall(embed_pattern, name):
                    self.embed_param_number += param.numel()
                elif "lm_head" in name:
                    self.lm_head_param_number += param.numel()
        

    def test_hook(self, name):
        """hook used for recording the time of gradient calculation, see comments on BACKWARD_VERBOSE for more details."""
        
        def func(x):
            if self.record_mark:
                self.backward_start_time = time.time()          
                self.record_mark = False
                relative_time = 0.
            else:
                relative_time = time.time() - self.backward_start_time
            if any(p_name in name for p_name in self.active_param_prefixs):
                print(f"param: {name:<50} relative time: {relative_time}")
            
            iterator = self.named_parameters_list
                
            for n, p in iterator:
                
                if p.requires_grad and p.grad is not None:
                    print("parameter name: ", n, "relative time", time.time() - self.backward_start_time)
                    
                    if (not any(p_name in n for p_name in self.active_param_prefixs)) and \
                        BACKWARD_VERBOSE == 2:
                        p.grad = None
                    
                    if len(self.ordered_named_params) < self.param_num:
                        self.ordered_named_params.append((n, p))
                    # break since for each step only one parameter's grad is updated
                    break
            return x
        
        return func

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        return self.base_optimizer.load_state_dict(state_dict)
    
    def _update_lr(self):
        # Make sure the learning rate of the base_optimizer is consistent with the BlockOptimizer
        for group in self.base_optimizer.param_groups:
            group["lr"] = self.param_groups[0]["lr"]
        
    def step(self, *args, **kwargs) -> None:
        self.record_mark = True
        # print("start")
        # print("finish")

        
        self._update_lr()
        self._grad_to_hp()
        self.base_optimizer.step(*args, **kwargs)
        self.calculate_grad_norm_for_each_layer()
        self._clean_hp_grad()

        

        self._update_param()
        self.global_step += 1


        torch.cuda.empty_cache()
        if (self.global_step + 1) % self.bcd_interval_steps == 0 or (self.bcd_order == "grad_distribution" and self.global_step <= self.total_layers):
            self.update_trainable_params()

    def _clean_hp_grad(self) -> None:
        """Clean the gradients of the high precision parameters."""
        for hp_param in self.param_idx2hp.values():
            hp_param.grad = None

    def _update_param(self) -> None:
        """Update the low precision parameters with the values of the high precision parameters."""
        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            lp_param.data.copy_(hp_param.to(lp_param.dtype).data)

    def _grad_to_hp(self, clear_lp_grads: bool = True) -> None:
        """
        Convert the gradients of the low precision parameters to high precision and calculate the gradient norm.

        Args:
            clear_lp_grads (bool, optional): Whether to clear the gradients of the low precision parameters. Defaults to True.
        """
        grad_norm = 0.0
        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            assert lp_param.grad is not None, "The low precision parameter's gradient is None."
            hp_param.grad = lp_param.grad.float()

            if clear_lp_grads:
                lp_param.grad = None

    def calculate_grad_norm_for_each_layer(self) :
        embed_pattern = r'.*embed[^.]*\.'
        layer_pattern = r'.*layers.[^.]*\.'
        import re

        self.current_grad_norms = [[] for _ in range(self.total_layers)]
        self.t_grad_norms = [[] for _ in range(self.total_layers)]
        for name, param in self.named_parameters_list:
            if not param.requires_grad  : 
                continue
            hp = self.lp_to_hp[param]
            grad = (param.to(dtype=torch.float32) - hp).to(torch.float16)
            current_grad_norm = torch.norm(grad).to(torch.float32).item()
            del grad
            t_grad = hp.grad
            t_gradnorm = torch.norm(t_grad)
            del t_grad
            del hp
            # state = self.base_optimizer.state[hp]
            # exp_avg = state['exp_avg']
            # exp_avg_sq = state["exp_avg_sq"]
            # op_step = state["step"]
            
            
            
            is_layer_param = False
            for i in range(self.total_layers) :
                if(self.block_prefix_list[i][0] in name) :
                    # self.current_grad_norms[i].append(current_grad_norm)
                    self.current_grad_norms[i].append(t_gradnorm)
                    self.t_grad_norms[i].append(t_gradnorm)
                    is_layer_param = True
                    break
            if not is_layer_param :
                if re.findall(embed_pattern, name):
                    self.embed_grad_norm_calculated_times += 1
                    self.embed_grad_norm += current_grad_norm
                    self.avg_embed_grad_norm = self.grad_beta * self.avg_embed_grad_norm + (1.0 - self.grad_beta) * current_grad_norm
                    self.avg_embed_grad_norm_hat = self.avg_embed_grad_norm / (1.0 - self.embed_grad_beta_multiple)
                    self.embed_grad_beta_multiple *= self.grad_beta
                    # print(f"embed : current_grad_norm={current_grad_norm} avg_grad_norms={self.avg_embed_grad_norm} param_number={self.embed_param_number}")
                elif "lm_head" in name:
                    self.lm_head_grad_norm_calculated_times += 1
                    self.lm_head_grad_norm += current_grad_norm
                    # print(f"lm_head_grad_norm={self.lm_head_grad_norm} lm_head_grad_norm_calculated_times = {self.lm_head_grad_norm_calculated_times}")
                    self.avg_lm_head_grad_norm = self.grad_beta * self.avg_lm_head_grad_norm + (1.0 - self.grad_beta) * current_grad_norm
                    self.avg_lm_head_grad_norm_hat = self.avg_lm_head_grad_norm / (1.0 - self.lm_head_grad_beta_multiple)
                    self.lm_head_grad_beta_multiple *= self.grad_beta
                    # print(f"lm_head : current_grad_norm={current_grad_norm} avg_grad_norms={self.avg_lm_head_grad_norm} param_number={self.lm_head_param_number}")
        
        for i in range(self.total_layers) :
            if len(self.current_grad_norms[i]) == 0 :
                continue
            self.grad_norms_calculated_times[i] += 1
            current_grad_norm = torch.norm(torch.tensor(self.current_grad_norms[i])).to(torch.float32).item()
            t_gradnorm_layer = torch.norm(torch.tensor(self.t_grad_norms[i]))
            wandb.log({f"layer{i} grad norm": current_grad_norm})
            print(f"layer{i} grad norm: {current_grad_norm}")
            print(f"T_gradnorm is: {t_gradnorm_layer}")
            self.grad_norms[i] += current_grad_norm
            self.avg_grad_norms[i] = self.avg_grad_norms[i]*self.grad_beta + (1.0 - self.grad_beta) * current_grad_norm
            self.avg_grad_norms_hat[i] = self.avg_grad_norms[i] / (1.0 - self.layers_grad_beta_multiple[i])
            self.layers_grad_beta_multiple[i] *= self.grad_beta
            self.last_grad_norms[i] += current_grad_norm
            # print(f"layer {i} : current_grad_norm={current_grad_norm} avg_grad_norms={self.avg_grad_norms[i]} param_number={self.layers_param_number[i]}")

            # print(f"torch norm calculate -- {name}: {torch.norm(param.grad)}")
        # print(self.avg_grad_norms)

    def weighted_sample_without_replacement(self, population, weights, k):
        
        population = list(population)
        weights = list(weights)

        selected = []
        for _ in range(k):
            
            chosen = random.choices(population, weights=weights, k=1)[0]
            selected.append(chosen)

            index = population.index(chosen)
            del population[index]
            del weights[index]

        return selected
    
    def optimizer_state_scale(self, x) :
        x_mx_val = torch.max(torch.abs(x)).to(dtype=torch.float32)
        scale_factor = ((2**7) - 1) / x_mx_val
        x_quantized = torch.round(x * scale_factor).to(dtype=torch.int8)
        return x_quantized, scale_factor
    
    def optimizer_state_unscale(self, x, scale_factor) :
        x = (x.to(dtype=torch.float32) / scale_factor).to(dtype=torch.float32)
        return x

    def update_trainable_params(self, verbose: Optional[int] = None) -> None:
        """
        Update the trainable parameters based on the current block index and the specified verbosity level.

        Args:
            verbose (Optional[int], optional): The verbosity level for printing information. Defaults to None.
        """
        self.param_idx2lp = {}
        self.param_idx2hp = {}
        # print("before:")
        # print(self.base_optimizer.state)

        # offload old optimizer state
        # for p, state in self.base_optimizer.state :
        #     if p 
        if self.offload_optimizer_state :
            if self.offload_rank > 0 :
                for n, p in self.named_parameters_list :

                    if p not in self.lp_to_hp :
                        continue
                    hp = self.lp_to_hp[p]
                    if not p.requires_grad or (hp not in self.base_optimizer.state) or (len(self.base_optimizer.state[hp]) == 0) :
                        continue
                    
                    for key, value in self.base_optimizer.state[hp].items() :
                        # do svd to value
                        if key == 'step' or len(value.shape) < 2 :
                            self.base_optimizer.state[p][key] = self.base_optimizer.state[hp][key]
                        else :
                            print(n, key, value.shape, flush=True)
                            U, S, Vh = torch.linalg.svd(value, full_matrices=False)
                            U = U[:, :self.offload_rank]
                            S = torch.diag(S[:self.offload_rank]) 
                            Vh = Vh[:self.offload_rank, :]
                            W1 = (U @ S).to('cpu')
                            del U, S
                            W2 = Vh.to('cpu')
                            del Vh
                            self.base_optimizer.state[p][key] = (W1, W2)
                            del W1, W2
                            
                    del self.base_optimizer.state[hp]
                    del hp
                    del self.lp_to_hp[p]
            elif self.offload_quantization_bit > 0 :
                for n, p in self.named_parameters_list :
                    # if p.requires_grad:
                    #     print(n, p.requires_grad, p in self.base_optimizer.state)
                    #     if p in self.base_optimizer.state :
                    #         print(len(self.base_optimizer.state[p]))
                    if p not in self.lp_to_hp :
                        continue
                    hp = self.lp_to_hp[p]
                    if not p.requires_grad or (hp not in self.base_optimizer.state) or (len(self.base_optimizer.state[hp]) == 0) :
                        continue
                    
                    for key, value in self.base_optimizer.state[hp].items() :
                        # print(n, key, value.shape)
                        if key == 'step'  :
                            self.base_optimizer.state[p][key] = self.base_optimizer.state[hp][key]
                        else :
                            self.base_optimizer.state[p][key] = value.to(dtype=torch.float16).to('cpu')
                    del self.base_optimizer.state[hp]
                    del hp
                    del self.lp_to_hp[p]
                # for n, p in self.named_parameters_list :
                #     # if p.requires_grad:
                #     #     print(n, p.requires_grad, p in self.base_optimizer.state)
                #     #     if p in self.base_optimizer.state :
                #     #         print(len(self.base_optimizer.state[p]))
                #     if p not in self.lp_to_hp :
                #         continue
                #     hp = self.lp_to_hp[p]
                #     if not p.requires_grad or (hp not in self.base_optimizer.state) or (len(self.base_optimizer.state[hp]) == 0) :
                #         continue
                    
                #     for key, value in self.base_optimizer.state[hp].items() :
                #         # print(n, key, value.shape)
                #         if key == 'step'  :
                #             self.base_optimizer.state[p][key] = self.base_optimizer.state[hp][key]
                #         else :
                #             value, scale_factor = self.optimizer_state_scale(value)
                #             value = value.to('cpu')
                #             self.base_optimizer.state[p][key] = (value, scale_factor)
                #     del self.base_optimizer.state[hp]
                #     del hp
                #     del self.lp_to_hp[p]
            else :
                for n, p in self.named_parameters_list :
                    # if p.requires_grad:
                    #     print(n, p.requires_grad, p in self.base_optimizer.state)
                    #     if p in self.base_optimizer.state :
                    #         print(len(self.base_optimizer.state[p]))
                    if p not in self.lp_to_hp :
                        continue
                    hp = self.lp_to_hp[p]
                    if not p.requires_grad or (hp not in self.base_optimizer.state) or (len(self.base_optimizer.state[hp]) == 0) :
                        continue
                    
                    for key, value in self.base_optimizer.state[hp].items() :
                        # print(n, key, value.shape)
                        self.base_optimizer.state[p][key] = self.base_optimizer.state[hp][key].to('cpu')
                    del self.base_optimizer.state[hp]
                    del hp
                    del self.lp_to_hp[p]
        else :
            self.lp_to_hp = {}
        # print("after:")

        # print(self.base_optimizer.state)
        if verbose is None:
            verbose = self.verbose
        if self.bcd_order == "random" :
            self.active_layers_indices = np.random.choice(range(self.total_layers), self.bcd_activated_layers, replace=False)
        elif self.bcd_order == "ascending" :
            if len(self.active_layers_indices):
                st = self.active_layers_indices[-1] + 1
                self.active_layers_indices = []
                for i in range(st, st + self.bcd_activated_layers) :
                    self.active_layers_indices.append(i % self.total_layers)
            else :
                self.active_layers_indices = [i for i in range(self.bcd_activated_layers)]
        elif self.bcd_order == "descending" :
            if len(self.active_layers_indices):
                st = self.active_layers_indices[-1] - 1
                self.active_layers_indices = []
                for i in range(st, st - self.bcd_activated_layers, -1) :
                    self.active_layers_indices.append((i + self.total_layers) % self.total_layers)
            else :
                self.active_layers_indices = [i for i in range(self.bcd_activated_layers)]
                # print(self.active_layers_indices)
                self.active_layers_indices.reverse()
        elif self.bcd_order == "min_grad" :
            
            self.active_layers_indices = sorted(range(len(self.last_grad_norms)), key=lambda i: self.last_grad_norms[i], reverse=True)[:self.bcd_activated_layers]
            for i in self. active_layers_indices :
                self.last_grad_norms[i] = 0
            
            # self.active_layers_indices = [10]
            
            print(f"Min grad method choose layers: f{self.active_layers_indices}")
            # print(self.active_layers_indices)
        elif self.bcd_order == "shuffling" :
            # print("test")
            if len(self.permutation) < self.bcd_activated_layers:
                self.permutation += np.random.permutation(self.total_layers).tolist()
            self.active_layers_indices = self.permutation[0: self.bcd_activated_layers]
            self.permutation = self.permutation[self.bcd_activated_layers:]
        elif self.bcd_order == "grad_distribution":
            # sample an layer from self.grad_norms[]

            if self.global_step < self.total_layers:
                # warm up to get initial grad norms
                self.active_layers_indices = [self.global_step]
            else :
                layer_selection_probabilities = self.avg_grad_norms_hat.copy()

                for i in range(len(layer_selection_probabilities)) :
                    layer_selection_probabilities[i] /= ((self.layer_selected_times[i]) ** self.grad_importance_exp)
                assert len(layer_selection_probabilities) == self.total_layers
                print(f"before divide: {self.avg_grad_norms_hat}\n")
                print(f"layer_selection_probabilities: {layer_selection_probabilities}")
                # print(f"avg_grad_norms_hat: {self.avg_grad_norms_hat}")
                self.active_layers_indices = self.weighted_sample_without_replacement(population=range(len(layer_selection_probabilities)), weights=layer_selection_probabilities, k=self.bcd_activated_layers)
                # self.active_layers_indices = np.random.choice(
                #     a=len(layer_selection_probabilities), 
                #     size=self.bcd_activated_layers,
                #     replace=False, 
                #     p=layer_selection_probabilities 
                # ).tolist()
            # print(f"hahaha: {self.activate_layers_indices}")
        if self.only_layer > self.total_layers:
            raise RuntimeError('The indicated only_layer number > the number of all layers')


        if self.only_layer != -1:
            self.active_layers_indices = [self.only_layer]
        print(f"Activating layers at indices: {self.active_layers_indices} for the next steps.", flush=True)
        
        for i in self.active_layers_indices :
            self.layer_selected_times[i] += 1
        print(f"Layer selected times: {self.layer_selected_times}")

        # self.active_param_prefixs = self.block_prefix_list[active_layers_indices]
        self.active_param_prefixs = []

        # print(self.block_prefix_list)
        # print(self.active_layers_indices)

        for i in self.active_layers_indices :
            self.active_param_prefixs.append(self.block_prefix_list[i][0])
        active_param_groups = [
            {
                "params": [],
                "weight_decay": self.param_groups[0]['weight_decay'],
                **self.defaults
            },
            {
                "params": [],
                "weight_decay": 0.0,
                **self.defaults
            },
        ]
        for i, (name, param) in enumerate(self.named_parameters_list):
            freezing_this_layer = False
            if not self.include_embedding_and_lm_head and not any(p in name for p in self.active_param_prefixs) :
                freezing_this_layer = True
            if self.include_embedding_and_lm_head and not any (p in name for p in self.other_params) and not any(p in name for p in self.active_param_prefixs) :
                freezing_this_layer = True
            if self.block_target_attn == False and ("att" in name):
                freezing_this_layer = True
            if self.block_target_mlp == False and (("fc" in name) or ("mlp" in name) or ("ffn" in name) or ("dense" in name and "att" not in name) or ("output" in name and "att" not in name)) :
                freezing_this_layer = True
            if self.block_target_non_linear == False and (("fc" not in name) and ("mlp" not in name) and ("dense" not in name) and ("output" not in name) and ("ffn" not in name) and ("att" not in name)) :
                freezing_this_layer = True
            if "classifier" in name :
                freezing_this_layer = False
            # print(name, freezing_this_layer)
            # if "lm_head" not in name :
            #     freezing_this_layer=True
            # else: freezing_this_layer = False

            if freezing_this_layer:
                param.requires_grad_(False)
                param.grad = None
                # print("NOT activated name: ", name)
            else:
                # print("activated name: ", name)
                if self.lora_mode and "lora" not in name:
                    continue

                param_hp = param.clone().float().detach().to(param.device)
                param.requires_grad_(True)
                param_hp.requires_grad = True
                
                self.param_idx2lp[i] = param
                self.param_idx2hp[i] = param_hp
                # if self.offload_optimizer_state:
                self.lp_to_hp[param] = param_hp
                
                if "bias" not in name and not isinstance(param, tuple(ALL_LAYERNORM_LAYERS)):
                    active_param_groups[0]['params'].append(param_hp)
                    # active_param_groups[0]['params'].append(param)
                else:
                    active_param_groups[1]['params'].append(param_hp)
                    # active_param_groups[1]['params'].append(param)
                
                if verbose >= 2:
                    print(name)

        self.base_optimizer.param_groups = active_param_groups
        # for param in self.base_optimizer.param_groups[0]["params"]:
        #     # print(f"{param.dtype}")
        #     if(param.grad) :
        #         print(f"gradient: {param.grad.dtype}")
        
        import gc
        gc.collect()
        
        # load old optimizer state from cpu
        if self.offload_optimizer_state :
            if self.offload_rank > 0 :
                for n, p in self.named_parameters_list :
                    if p not in self.lp_to_hp :
                        continue    
                    hp = self.lp_to_hp[p]
                    if not p.requires_grad or (p not in self.base_optimizer.state) :
                        continue
                    self.base_optimizer.state[hp] = {}
                    for key, value in self.base_optimizer.state[p].items() :
                        # print(n, key, value)
                        if not isinstance(value, tuple):
                            self.base_optimizer.state[hp][key] = self.base_optimizer.state[p][key].to(self.device)
                        else :
                            assert isinstance(self.base_optimizer.state[p][key], tuple)
                            W1 = self.base_optimizer.state[p][key][0].to(self.device)
                            W2 = self.base_optimizer.state[p][key][1].to(self.device)

                            self.base_optimizer.state[hp][key] = W1 @ W2
                            del W1, W2
                            assert self.base_optimizer.state[hp][key].device == self.device
                    del self.base_optimizer.state[p]
            elif self.offload_quantization_bit > 0 :
                for n, p in self.named_parameters_list :
                    if p not in self.lp_to_hp :
                        continue    
                    hp = self.lp_to_hp[p]
                    if not p.requires_grad or (p not in self.base_optimizer.state) :
                        continue
                    self.base_optimizer.state[hp] = {}
                    for key, value in self.base_optimizer.state[p].items() :
                        if key == 'step':
                            self.base_optimizer.state[hp][key] = self.base_optimizer.state[p][key]
                        # print(n, key, value)
                        else :
                            # value, scale_factor = value
                            self.base_optimizer.state[hp][key] = value.to(dtype=torch.float32).to(self.device)
                    del self.base_optimizer.state[p]
                # for n, p in self.named_parameters_list :
                #     if p not in self.lp_to_hp :
                #         continue    
                #     hp = self.lp_to_hp[p]
                #     if not p.requires_grad or (p not in self.base_optimizer.state) :
                #         continue
                #     self.base_optimizer.state[hp] = {}
                #     for key, value in self.base_optimizer.state[p].items() :
                #         if not isinstance(value, tuple):
                #             self.base_optimizer.state[hp][key] = self.base_optimizer.state[p][key].to(self.device)
                #         # print(n, key, value)
                #         else :
                #             value, scale_factor = value
                #             value = value.to(self.device)
                #             value = self.optimizer_state_unscale(value, scale_factor)
                #             self.base_optimizer.state[hp][key] = value
                #     del self.base_optimizer.state[p]
            else :
                for n, p in self.named_parameters_list :
                    if p not in self.lp_to_hp :
                        continue    
                    hp = self.lp_to_hp[p]
                    if not p.requires_grad or (p not in self.base_optimizer.state) :
                        continue
                    self.base_optimizer.state[hp] = {}
                    for key, value in self.base_optimizer.state[p].items() :
                        # print(n, key, value)
                        self.base_optimizer.state[hp][key] = self.base_optimizer.state[p][key].to(self.device)
                    del self.base_optimizer.state[p]

        else :
            # Clean the optimizer state
            self.base_optimizer.state = defaultdict(lambda: {})
        
        # for n, p in self.named_parameters_list :
        #     if p not in self.lp_to_hp:
        #         continue
        #     print(f"{n}:")
        #     hp = self.lp_to_hp[p]
        #     for key, value in self.base_optimizer.state[hp].items() :
        #         # self.base_optimizer.state[hp][key] = value.to(dtype=torch.float16).to(dtype=torch.float32)
        #         delta = value - value.to(dtype=torch.float16).to(dtype=torch.float32)
        #         # scaled_value, scale_factor = self.optimizer_state_scale(value)
        #         # dequantized = self.optimizer_state_unscale(scaled_value, scale_factor)
        #         # delta = torch.abs(value - dequantized)
        #         print(n, key, f"ERR[{float(torch.sum(torch.abs(delta)) /torch.sum(torch.abs(value)))}]", torch.norm(delta).data, torch.norm(value).data, torch.sum(torch.abs(delta)).data, torch.sum(torch.abs(value)).data)
        #         # print(n, p.requires_grad, key, value.device, value.dtype, value.shape)

        # print("end: ")
        # print(self.base_optimizer.state)
        trainable = 0
        untranable = 0
        for n, p in self.named_parameters_list:
            # if p.requires_grad: 
                # print(f"{n} {p.requires_grad} {p.numel()}")
            if p.requires_grad:
                trainable += p.numel()
            else :
                untranable += p.numel()
        print(f"{trainable}/{trainable+untranable}  =  {trainable/(trainable+untranable)}")

        wandb.log({"trainable_param_ratio":  trainable/(trainable+untranable)})




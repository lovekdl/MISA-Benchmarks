import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union
import wandb
from dataclasses import dataclass, field
from src.custom_trainer import OurTrainer
from src.args import CustomizedTrainingArguments
from utils.callbacks import MemoryTimeConsumptionCallback, CommonsenseReasoningEvaluateCallback
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import (  # noqa: E402
    LoraConfig,
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

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the dataset to use ."}
    )
    train_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the dataset to train ."}
    )
    test_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the dataset to test ."}
    )
    task_name: Optional[str] = field(
        default=None, metadata={"help": "Task name."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    max_tokens_per_batch: Optional[int] = field(
        default=0,
        metadata={
            "help": "dynamic batching. Override batch size when larger than 0"
        },
    ),


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    load_adapter_config: bool = field(
        metadata={"help": "Whether the model is saved normally of with adapter config separately"}
    )

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_load_path_adapter: str = field(
        default="",
        metadata={"help": ""}
    )
    ### INSERTED BELOW
    # model_load_path_second: str = field(
    #    default="",
    #    metadata={"help": ""}
    # )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )

    # prefix-tuning parameters
    add_enc_prefix: bool = field(
        default=False,
        metadata={"help": "Whether use prefix tuning"},
    )
    add_dec_prefix: bool = field(
        default=False,
        metadata={"help": "Whether use prefix tuning"},
    )
    add_cross_prefix: bool = field(
        default=False,
        metadata={"help": "Whether use prefix tuning"},
    )
    prefix_len: Optional[int] = field(
        default=10,
        metadata={"help": "length of prefix tokens"},
    )
    mid_dim: Optional[int] = field(
        default=512,
        metadata={"help": "dim of middle layer"},
    )
    # bitfit parameters
    tune_bias: bool = field(
        default=False,
        metadata={"help": "Whether tune bias terms"},
    )
    # LoRA parameters
    lora: bool = field(
        default=False,
        metadata={"help": "Whether to use lora finetuning"},
    )
    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "rank of lora"},
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "scaling = alpha / r"},
    )
    lora_target: Optional[str] = field(
        default='all',
        metadata={"help": "lora target"},
    )
    use_dora:Optional[bool] = field(
        default=False,
        metadata={"help":"use dora finetuning."}
    )
    drop_first_layers: Optional[int] = field(
        default=0,
        metadata={
            "help": "drop first k layers, work for both prefix and adapter, freeze transformer layers if fine-tuning"},
    )
    drop_first_adapter_layers: Optional[int] = field(
        default=0,
        metadata={"help": "drop first k adapter layers"},
    )
    drop_first_prefix_layers_enc: Optional[int] = field(
        default=0,
        metadata={"help": "drop first k prefix layers"},
    )
    drop_first_prefix_layers_dec: Optional[int] = field(
        default=0,
        metadata={"help": "drop first k prefix layers"},
    )
    drop_first_prefix_layers_cross: Optional[int] = field(
        default=0,
        metadata={"help": "drop first k prefix layers"},
    )
    add_adapter_gate: bool = field(
        default=True,
        metadata={"help": "add a gate to the adapter"},
    )
    add_prefix_gate: bool = field(
        default=True,
        metadata={"help": "add a gate to the prefix"},
    )
    add_lora_gate: bool = field(
        default=True,
        metadata={"help": "add a gate to the lora"},
    )
    add_central_gate: bool = field(
        default=False,
        metadata={"help": "add a shared gate"},
    )




@dataclass
class GenerationArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    min_length: Optional[int] = field(
        default=10,
        metadata={
            "help": "minimal generation length"
        },
    )

    max_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "max generation length"
        },
    )

    num_beams: Optional[int] = field(
        default=5,
        metadata={
            "help": "minimal generation length"
        },
    )

    no_repeat_ngram_size: Optional[int] = field(
        default=0,
        metadata={
            "help": "minimal generation length"
        },
    )

    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "length penalty"
        },
    )


@dataclass
class TuneArguments:
    attn_mode: Optional[str] = field(
        default="none",
        metadata={
            "choices": ["prefix", "prefix_nomlp",
                        "none", "bitfit", "lora", "adapter",
                        "prompt_tuning"], \
 \
            "help": "config for attention, none to disable; \
                prefix: mlp reparameterization to output prefix P; \
                prefix_nomlp: prefix P as learned params; \
                adapter: adapter mode; \
                bitfit: the bitfit baseline; \
                lora: the lora baseline; \
                prompt_tuning: the prompt tuning baseline",
        },
    )

    attn_option: Optional[str] = field(
        default="concat",
        metadata={
            "choices": ["none",
                        "concat",
                        "cross_attn",
                        "cross_attn_noln",
                        "cross_attn_relu",
                        "parallel",
                        "sequential",
                        ], \
 \
            "help": "specific attn configs; \
                concat: concat prefix to self, this is prefix tuning baseline; \
                cross_attn_noln: prefix tuning with vanilla add composition (instead of gated add), \
                    need to be used together with 'attn_composition=add'; \
                cross_attn: cross_attn_noln plus a layernorm layer \
                cross_attn_relu: basically multi-head adapter, need to be used under 'prefix' mode; \
                parallel: parallel insertion form; need to be used under 'adapter' mode; \
                sequential: sequential insertion form; need to be used under 'adapter' mode;",

        },
    )

    attn_composition: Optional[str] = field(
        default="add",
        metadata={
            "choices": ["add", "gate_add"],
            "help": "the composition function \
                add: vanilla adding; \
                gate_add: gated adding like prefix tuning"
        },
    )

    ffn_mode: Optional[str] = field(
        default="none",
        metadata={
            "choices": ["adapter", "none", "lora"],

            "help": "config for ffn, none to disable; \
            adapter: adapter mode; \
            lora: the lora baseline",
        },
    )

    ffn_option: Optional[str] = field(
        default="none",
        metadata={
            "choices": ["parallel", "sequential", "pfeiffer", "none"], \
 \
            "help": "specific ffn configs; \
                parallel: parallel insertion form; \
                sequential: sequential insertion form; \
                pfeiffer: the Pfeiffer adapter config"
        },
    )

    ffn_adapter_layernorm_option: Optional[str] = field(
        default="in",
        metadata={
            "choices": ["in", "out", "none"],
            "help": "ffn adapter layernorm options; \
                none: no layernorm; \
                in: layernorm applied to input; \
                out: layernorm applied to output"
        },
    )

    ffn_adapter_init_option: Optional[str] = field(
        default="bert",
        metadata={
            "choices": ["bert", "lora"],
            "help": "ffn adapter option"
        },
    )

    ffn_adapter_scalar: Optional[str] = field(
        default="1",
        metadata={
            "help": "the scaling hyperparam for scaled adding composition; \
                set to 'learnable_scalar' to learn this as a parameter"
        },
    )

    mid_dim: Optional[int] = field(
        default=800,
        metadata={
            "help": ""
        },
    )

    attn_bn: Optional[int] = field(
        default=200,
        metadata={
            "help": "the attention bottleneck dimension"
        },
    )

    ffn_bn: Optional[int] = field(
        default=-1,
        metadata={
            "help": "the ffn bottleneck dimension"
        },
    )

    prefix_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": ""
        },
    )

    unfreeze_params: Optional[str] = field(
        default="ef_",
        metadata={
            "help": "param names that contain the string will \
                be unfreezed, all other params will be freezed"
        },
    )

    load_path: Optional[str] = field(
        default="",
        metadata={
            "help": ""
        },
    )

    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={
            "help": "scaling: alpha / r"
        },
    )

    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "scaling: alpha / r"
        },
    )

    lora_init: Optional[str] = field(
        default="lora",
        metadata={
            "choices": ["bert", "lora"],
            "help": ""
        },
    )


@dataclass
class AdapterArguments:
    """
    The subset of arguments related to model training.
    """

    train_adapter: bool = field(default=False, metadata={"help": "Train an model instead of the full model."})
    load_adapter: Optional[str] = field(
        default="", metadata={"help": "Pre-trained model module to be loaded from Hub."}
    )
    adapter_config: Optional[str] = field(
        default="pfeiffer", metadata={"help": "Adapter configuration. Either an identifier or a path to a file."}
    )
    adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the model configuration."}
    )
    adapter_reduction_factor: Optional[int] = field(
        default=None, metadata={"help": "Override the reduction factor of the model configuration."}
    )
    language: Optional[str] = field(default=None, metadata={"help": "The training language, e.g. 'en' for English."})


@dataclass
class MultiLingAdapterArguments(AdapterArguments):
    """
    Arguemnts related to model training, extended by arguments for multilingual setups.
    """

    load_lang_adapter: Optional[str] = field(
        default=None, metadata={"help": "Pre-trained language model module to be loaded from Hub."}
    )
    lang_adapter_config: Optional[str] = field(
        default=None, metadata={"help": "Language model configuration. Either an identifier or a path to a file."}
    )
    lang_adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the language model configuration."}
    )
    lang_adapter_reduction_factor: Optional[int] = field(
        default=None, metadata={"help": "Override the reduction factor of the language model configuration."}
    )

class SparseUpdateTrainer(Trainer):
    def __init__(self, *args, mask, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)

        # mask out the gradients
        for name, params in self.model.named_parameters():

            device = params.device
            if name in self.mask:
                self.mask[name] = self.mask[name].to(device)
                params.grad.data.copy_(params.grad.data * self.mask[name].data)

        return loss




def generate_tag(model_args, data_args, training_args) :
    run_name = ''
    args = training_args
    if args.optimizer == "bcd-optimizer" :
        run_name = "BCD"
        if model_args.lora:
            run_name = "Mix_lora_bcd"
            training_args.mix_lora = True
        if args.only_layer != -1:
            run_name += f"-only_layer{args.only_layer}"
            run_name += f"-TASK[{data_args.task_name}]-{model_args.model_name_or_path}-EPOCH{args.num_train_epochs}-{args.bcd_base_optimizer}-LR{args.learning_rate}"
        else :
            run_name += f"-{args.bcd_activated_layers}layers-interval{args.bcd_interval_steps}-order{args.bcd_update_order}-TASK[{data_args.task_name}]-{model_args.model_name_or_path}-EPOCH{args.num_train_epochs}-{args.bcd_base_optimizer}-LR{args.learning_rate}"
        if args.offload_optimizer_state:
            run_name += f"—OffloadOptim"
        run_name += f"Gran[{args.granularity}]"
        if args.granularity == 'module' :
            run_name += f"[{args.param_ratio_limit}]"

        targets = []
        if args.bcd_target_attn:
            targets.append("attn")
        if args.bcd_target_mlp :
            targets.append("mlp")
        run_name += "_target_" + '_'.join(targets) 
    elif model_args.lora:
        run_name += f"LORA-rank{model_args.lora_r}-lora_target[{model_args.lora_target}]-alpha{model_args.lora_alpha}-TASK[{data_args.task_name}]-{model_args.model_name_or_path}-EPOCH{args.num_train_epochs}-OPTIM[{args.optimizer}]-LR{args.learning_rate}"
    else :
        run_name +=f"TASK[{data_args.task_name}]-{model_args.model_name_or_path}-EPOCH{args.num_train_epochs}-OPTIM[{args.optimizer}]-LR{args.learning_rate}"
    return run_name



def main() :
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomizedTrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.task_name is None:
        data_args.task_name = data_args.dataset_path

    run_name = generate_tag(model_args, data_args, training_args)

    wandb.init(project='test_project', name=run_name)


    set_seed(training_args.seed)
    # global times
    times = 0
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map='auto',
        config=config,
        torch_dtype=torch.float16,
    )
    print(torch.cuda.max_memory_allocated()/(1024**3))
    if ("llama" in model_args.model_name_or_path.lower()) or ("mistral" in model_args.model_name_or_path.lower()):
            # LLaMA padding token
            tokenizer.pad_token_id = 0
    # print(torch.cuda.max_memory_allocated()/  (1024**3))
    # exit(0)

    if model_args.lora:
        def find_all_linear_modules(model: "PreTrainedModel") -> list[str]:
            r"""
            Finds all available modules to apply lora or galore.
            """
            quantization_method = getattr(model, "quantization_method", None)
            if quantization_method is None:
                linear_cls = torch.nn.Linear
            else:
                raise ValueError("Finding linear modules for {} models is not supported.".format(quantization_method))

            output_layer_names = ["lm_head"]
            if model.config.model_type == "chatglm":
                output_layer_names.append("output_layer")
            elif model.config.model_type == "internlm2":
                output_layer_names.append("output")

            module_names = set()
            for name, module in model.named_modules():
                # print(name)
                if isinstance(module, linear_cls) and not any(output_layer in name for output_layer in output_layer_names):
                    module_names.add(name.split(".")[-1])
            module_names.add("lm_head")
            module_names.add("embed_tokens")
            print("Found modules: {}".format(",".join(module_names)))
            return list(module_names)
        

        lora_target = model_args.lora_target
        if isinstance(lora_target, str):
            lora_target = [name.strip() for name in lora_target.split(",")]
        if lora_target[0] == "all":
            lora_target = find_all_linear_modules(model)
        # logger.info(f"All linear modules: {find_all_linear_modules(model)}")
        print(f"lora_target : {lora_target}")
        from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model
        peft_kwargs = {
            "r": model_args.lora_r,
            "target_modules": lora_target,
            "lora_alpha": model_args.lora_alpha,
            "lora_dropout": 0.05,
        }
        lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                use_dora=model_args.use_dora,
                **peft_kwargs,
            )
        model = get_peft_model(model, lora_config)
            # if model.config.model_type == "llama":
            #     tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)
            # else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
        
    tokenizer.padding_side = "left" 

    trainable = 0
    ntrainable = 0
    for n,p in model.named_parameters() :
        print(n, p.requires_grad, p.numel())
        if p.requires_grad :
            trainable += p.numel()
        else: ntrainable += p.numel()
    for i in range(10):
        print(f"Trainable parameters: {trainable} / {ntrainable+trainable}")
    
    wandb.log({"trainable_param_ratio": trainable / (ntrainable+trainable)})
    
    def tokenize( prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=data_args.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        # global times
        # if times < 5:
        #     times += 1
        #     print(f"{prompt}\n{result}\n\n")
        #     print(len(result['input_ids']), len(result['attention_mask']))
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < data_args.max_seq_length
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in model_args.model_name_or_path:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in model_args.model_name_or_path:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point, with_answer=True):
        if with_answer:
            full_prompt = generate_prompt(data_point)
            tokenized_full_prompt = tokenize(full_prompt)

            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                    -100
                                                ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        else :
            
            full_prompt = generate_prompt(data_point, with_answer=False)
            tokenized_full_prompt = tokenize(full_prompt, add_eos_token=False)

        return tokenized_full_prompt


    def generate_prompt(data_point, with_answer=True):
        # sorry about the formatting disaster gotta move fast
        if with_answer :
            if data_point["input"]:
                return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}""" # noqa: E501
            else:
                return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}""" # noqa: E501
        else :
            if data_point["input"]:
                return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:\n""" # noqa: E501
            else:
                return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{data_point["instruction"]}

### Response:\n""" # noqa: E501



            

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable_params: {trainable_params}, total_params: {total_params}")

    # load train dataset
    if data_args.dataset_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_args.dataset_path)
    else:
        data = load_dataset(data_args.dataset_path)
    train_data = data["train"].shuffle(seed=training_args.seed).map(generate_and_tokenize_prompt)
    
    # load test dataset 
    test_names = ['AQuA', 'mawps', 'GSM8K', 'SVAMP']
    test_data = {}
    for n in test_names:
        test_d = load_dataset(f'datasets/{n}')
        test_data[n] = test_d["test"].shuffle(seed=training_args.seed).map(
            generate_and_tokenize_prompt,
            fn_kwargs={"with_answer":False}
        )



    do_eval = False
    eval_strategy = training_args.eval_strategy
    if training_args.do_eval:
        do_eval = True
        training_args.do_eval = 'no'
        training_args.eval_strategy = 'no'
    
    print(train_data)
    training_args.per_device_eval_batch_size = 1
    trainer = OurTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=None,
        # compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        )
    from utils.callbacks import MemoryTimeConsumptionCallback, MathReasoningEvaluateCallback
    trainer.add_callback(MemoryTimeConsumptionCallback(trainer))
    if do_eval :  
            trainer.add_callback(MathReasoningEvaluateCallback(
                trainer=trainer,
                model=model,
                tokenizer=tokenizer,
                test_dataset=test_data,
                dataset_name=data_args.task_name,
                eval_strategy=eval_strategy,
                args=training_args,
                ))


    train_result = trainer.train()
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_data)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_data))

    # trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()
import wandb
import transformers
import torch
import copy
import time
import json
from tqdm import tqdm
from transformers import GenerationConfig
import re
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class MemoryTimeConsumptionCallback(transformers.TrainerCallback) :
    def __init__(self, trainer) -> None:
        super().__init__()
        self.trainer = trainer
    def on_step_begin(self, args, state, control, **kwargs):
        self.start = time.time()
    def on_step_end(self, args, state, control, **kwargs):
        self.end = time.time()
        train_step_duration = self.end - self.start
        max_memory_allocated = 0
        for device_id in range(torch.cuda.device_count()):
            # this is not accurate since max memory does not happen simultaneously across all devices
            max_memory_allocated += torch.cuda.max_memory_allocated(device_id)
        
        flag = control.should_log
        self.trainer.log({"peak_mem": max_memory_allocated / 1024 ** 3,
                    "step_consumption": train_step_duration * 1000})
        control.should_log = flag

        wandb.log({"peak_mem": max_memory_allocated / 1024 ** 3,
                    "step_consumption": train_step_duration * 1000})


class CommonsenseReasoningEvaluateCallback(transformers.TrainerCallback) :
    def __init__(self, trainer, model, tokenizer, test_dataset, dataset_name, eval_strategy, args) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.dataset_name = dataset_name
        self.args = args
        self.trainer = trainer 
        self.eval_strategy = eval_strategy

        self.save_file =  f"{self.args.output_dir}/eval.json"

        self.batches = self.create_batch(test_dataset, args.per_device_eval_batch_size if args.per_device_eval_batch_size else args.per_device_train_batch_size )
        self.epoch = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.eval_strategy != 'epoch' :
            return
        self.epoch += 1
        self.evaluate()
    # def on_train_begin(self, args, state, control, **kwargs):
    #     self.evaluate()
    #     exit(0)
    def on_step_end(self, args, state, control, **kwargs):

        # self.evaluate()
        if self.eval_strategy != 'steps' :
            return
        if state.global_step % self.args.eval_steps :
            return
        self.evaluate()
    
    def evaluate(self) :
        batches = self.batches
        total = len(batches)
        correct = 0
        current = 0
        output_data = []
        pbar = tqdm(total=total)

        for idx, batch in enumerate(batches):
            
            outputs = self.batch_generate(batch)
            data_answer = batch['answer']
            l = len(data_answer)
            current += l

            for i in range(len(data_answer)):
                label = data_answer[i]
                output = outputs[i]
                flag = False
                predict = self.extract_answer(output)
                if label == predict:
                    correct += 1
                    flag = True
                new_data = {'label':label}
                new_data['output_pred'] = output
                new_data['pred'] = predict
                new_data['flag'] = flag
                output_data.append(new_data)
                # print(data["instruction"])
                # print(output)
                
                print('prediction:', predict)
                print('label:', label)
            print('---------------')
            print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
            print('---------------')
            with open(self.save_file, 'w+') as f:
                json.dump(output_data, f, indent=4)
            pbar.update(1)

        self.trainer.log({"eval_accuracy": correct / current ,})
        wandb.log({"eval_accuracy": correct / current ,})
            
        pbar.close()
        print('\n')
        print('test finished')

    def batch_generate(
            self,
            data, 
            # temperature=0.1,
            # top_p=0.75,
            # top_k=40,
            # num_beams=4,
            max_new_tokens=64,
            **kwargs,
    ):
        # prompts = [generate_prompt(instruction, input) for instruction in instructions]
        # inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        # print(data)
        # print(data)
        # for x in data["input_ids"]:
        #     print(x, len(x))
        input_ids = data["input_ids"]
        encoded_input = self.tokenizer.pad(
                {"input_ids": input_ids}, 
                padding=True, 
                return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # print(len(input_ids))
        # generation_config = GenerationConfig(
        #     temperature=temperature,
        #     attention_mask=attention_mask,
        #     top_p=top_p,
        #     top_k=top_k,
        #     num_beams=num_beams,
        #     **kwargs,
        # )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                # generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = self.tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("### Response:")[1].strip() for o in outputs]
        return outputs


    def create_batch(self, dataset, batch_size):
        batches = []
        num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
        for i in range(num_batch):
            batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
            batches.append(batch)
        return batches

    def extract_answer(self, sentence: str) -> float:
        dataset = self.dataset_name
        if dataset == 'boolq':
            sentence_ = sentence.strip()
            pred_answers = re.findall(r'true|false', sentence_)
            if not pred_answers:
                return ""
            return pred_answers[0]
        elif dataset == 'piqa':
            sentence_ = sentence.strip()
            pred_answers = re.findall(r'solution1|solution2', sentence_)
            if not pred_answers:
                return ""
            return pred_answers[0]
        elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
            sentence_ = sentence.strip()
            pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
            if not pred_answers:
                return ""
            return pred_answers[0]
        elif dataset == 'hellaswag':
            sentence_ = sentence.strip()
            pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
            if not pred_answers:
                return ""
            return pred_answers[0]
        elif dataset == 'winogrande':
            sentence_ = sentence.strip()
            pred_answers = re.findall(r'option1|option2', sentence_)
            if not pred_answers:
                return ""
            return pred_answers[0]
            


class MathReasoningEvaluateCallback(transformers.TrainerCallback) :
    def __init__(self, trainer, model, tokenizer, test_dataset, dataset_name, eval_strategy, args) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.dataset_name = dataset_name
        self.args = args
        self.trainer = trainer 
        self.eval_strategy = eval_strategy

        self.save_file =  f"{self.args.output_dir}/eval.json"
        self.batches = {}
        self.epoch = 0
        for n, d in test_dataset.items() :

            self.batches[n] = self.create_batch(d, args.per_device_eval_batch_size if args.per_device_eval_batch_size else 1 )
        


    def on_epoch_end(self, args, state, control, **kwargs):
        if self.eval_strategy != 'epoch' :
            return
        self.epoch += 1 
        if self.epoch % 3 == 0:
            self.evaluate()
    # def on_train_begin(self, args, state, control, **kwargs):
    #     self.evaluate()
    #     exit(0)
    def on_step_end(self, args, state, control, **kwargs):

        # self.evaluate()
        if self.eval_strategy != 'steps' :
            return
        if state.global_step % self.args.eval_steps :
            return
        self.evaluate()
    
    def evaluate(self) :
        

        for n, data_batches  in self.batches.items():
            total = len(data_batches)
            correct = 0
            current = 0
            output_data = []
            pbar = tqdm(total=total)
            for idx, batch in enumerate(data_batches):
                
                outputs = self.batch_generate(batch)
                data_answer = batch['answer']
                l = len(data_answer)
                current += l

                for i in range(len(data_answer)):
                    label = data_answer[i]
                    output = outputs[i]
                    eps = 0.001
                    flag = False
                    if n.lower() in ['aqua']:
                        predict = self.extract_answer_letter(output)
                        if label == predict:
                            correct += 1
                            flag = True
                    else:
                        if isinstance(label, str):
                            label = float(label)
                        predict = self.extract_answer_number(n, output)
                        if abs(label - predict) <= eps:
                            correct += 1
                            flag = True

                    new_data = {'label':label}
                    new_data['output_pred'] = output
                    new_data['pred'] = predict
                    new_data['flag'] = flag
                    output_data.append(new_data)
                    # print(data["instruction"])
                    # print(output)
                    print('output:', output)
                    print('prediction:', predict)
                    print('label:', label)
                print('---------------')
                print(f'\rtest[{n}]:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
                print('---------------')
                with open(self.save_file, 'w+') as f:
                    json.dump(output_data, f, indent=4)
                pbar.update(1)

            self.trainer.log({f"{n}_accuracy": correct / current ,})
            wandb.log({f"{n}_accuracy": correct / current ,})
                
            pbar.close()
            print('\n')
            print(f'test{n} finished')

    def batch_generate(
            self,
            data, 
            # temperature=0.1,
            # top_p=0.75,
            # top_k=40,
            # num_beams=4,
            max_new_tokens=512,
            **kwargs,
    ):
        # prompts = [generate_prompt(instruction, input) for instruction in instructions]
        # inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        # print(data)
        # print(data)
        # for x in data["input_ids"]:
        #     print(x, len(x))
        input_ids = data["input_ids"]
        encoded_input = self.tokenizer.pad(
                {"input_ids": input_ids}, 
                padding=True, 
                return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # print(len(input_ids))
        # generation_config = GenerationConfig(
        #     temperature=temperature,
        #     attention_mask=attention_mask,
        #     top_p=top_p,
        #     top_k=top_k,
        #     num_beams=num_beams,
        #     **kwargs,
        # )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                # generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = self.tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("### Response:")[1].strip() for o in outputs]
        return outputs


    def create_batch(self, dataset, batch_size):
        batches = []
        num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
        for i in range(num_batch):
            batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
            batches.append(batch)
        return batches

    def extract_answer_number(self, n: str, sentence: str) -> float:
        dataset = n.lower()
        if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:
            sentence = sentence.replace(',', '')
            pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
            if not pred:
                return float('inf')
            pred_answer = float(pred[-1])
        else:
            raise NotImplementedError(' not support dataset: {}'.format(dataset))
        if isinstance(pred_answer, str):
            try:
                pred_answer = float(pred_answer)
            except ValueError as e:
                pred_answer = float('inf')
        return pred_answer


    def extract_answer_letter(self, sentence: str) -> str:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'A|B|C|D|E', sentence_)
        if pred_answers:
            if not pred_answers:
                return ''
            return pred_answers[0]
        else:
            return ''
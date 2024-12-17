from datasets import load_dataset
import json

data = load_dataset('openai/gsm8k', 'main')

def convert_to_json_format(example):
    question = example['question']
    answer_with_steps = example['answer']
    
    if '####' in answer_with_steps:
        final_answer = answer_with_steps.split('####')[-1].strip()
    else:
        final_answer = ""

    return {
        "instruction": question,
        "input": "",
        "output": answer_with_steps,
        "answer": final_answer
    }

formatted_data = [convert_to_json_format(example) for example in data['train']]

output_file = "./datasets/gsm8k/train.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=4)

print(f"数据已成功保存到 {output_file}")

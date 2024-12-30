from datasets import load_dataset


data = load_dataset("openai/gsm8k", "main")
# en = load_dataset("allenai/c4", "en", split="validation")
print(data)
# print(en)   
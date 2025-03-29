from datasets import load_dataset

# import evaluate
# dataset = load_dataset(
#             "lukaemon/mmlu", "high_school_computer_science", revision="refs/convert/parquet", trust_remote_code=True
#         )
dataset = load_dataset(
            "cais/mmlu", "high_school_computer_science", trust_remote_code=True
        )
print(dataset["dev"][0])
# data = load_dataset("/root/workspace/cthu1g4p420c73egakcg/BCD-Benchmarks/datasets/C4/en")
# print(data)
# print(en)   
# dataset = load_dataset(path="hgissbkh/super_glue", name="copa")
# print(dataset)

# metric = evaluate.load("super_glue", "copa")
# print(metric)


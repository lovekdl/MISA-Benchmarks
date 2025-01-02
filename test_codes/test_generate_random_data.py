import random

def generate_random_ids(seq_len: int, vocab_size: int = 32000) -> list:
    """
    生成指定序列长度的随机 input_ids，取值范围在 [0, vocab_size - 1] 之间。
    """
    return [random.randint(0, vocab_size - 1) for _ in range(seq_len)]

def generate_dataset(min_len=100, max_len=100, step=100, vocab_size=32000):
    """
    生成一个包含不同序列长度的随机数据集，每条数据的结构如下：
    {
        "sequence_length": seq_len,
        "input_ids": [...],
        "attention_mask": [...]
    }
    :param min_len: 序列长度下限
    :param max_len: 序列长度上限
    :param step: 递增步长
    :param vocab_size: 词表大小（决定 token id 的取值范围）
    :return: 数据集(list)，从小到大按序列长度排序
    """
    dataset = []
    for seq_len in range(min_len, max_len + 1, step):
        input_ids = generate_random_ids(seq_len, vocab_size)
        attention_mask = [1] * seq_len  # 在真实场景下，可根据需要改为更复杂的逻辑
        dataset.append({
            "sequence_length": seq_len,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
    return dataset

if __name__ == "__main__":
    data = generate_dataset(min_len=100, max_len=100, step=100, vocab_size=32000)
    
    # 打印前几条数据的示例，查看序列长度
    for item in data[:5]:
        print(f"Sequence length: {item['sequence_length']}")
        print(f"Input IDs sample: {item['input_ids'][:]}...")
        print(f"Attention Mask: {item['attention_mask'][:]}...\n")

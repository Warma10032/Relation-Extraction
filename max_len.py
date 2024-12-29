import json
from transformers import BertTokenizer


# 读取jsonl文件
def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


# 加载BERT分词器
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")


# 统计数据集文本长度
def get_sentence_lengths(data):
    sentence_lengths = []

    for sample in data:
        sentence = sample["sentence"]
        tokens = tokenizer.tokenize(sentence)  # 获取token
        sentence_lengths.append(len(tokens))  # 记录token数

    return sentence_lengths


# 计算最大、最小、平均长度
def calculate_statistics(lengths):
    max_len = max(lengths)
    min_len = min(lengths)
    avg_len = sum(lengths) / len(lengths)
    return max_len, min_len, avg_len


file_path = "./dataset/test.jsonl" 
train_data = read_jsonl(file_path)

# 获取句子长度
sentence_lengths = get_sentence_lengths(train_data)

# 计算统计信息
max_len, min_len, avg_len = calculate_statistics(sentence_lengths)

print(f"最大句子长度: {max_len}")
print(f"最小句子长度: {min_len}")
print(f"平均句子长度: {avg_len:.2f}")

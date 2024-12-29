# test.py
import torch
import json
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, classification_report
from data import RelationshipDataset


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def evaluate_model(model_path, test_data, text_build_type):
    # 配置参数
    MAX_LEN = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    # 准备测试数据
    test_dataset = RelationshipDataset(test_data, tokenizer, MAX_LEN, text_build_type)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    # 评估模式
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 计算micro指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="micro"
    )

    # 计算macro指标
    print(classification_report(true_labels, predictions, digits=4))

    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    test_data = read_jsonl("./dataset/test.jsonl")
    text_build_model = ["basic1", "basic2", "QA", "entity_marked1", "entity_marked2"]

    # 测试每个模型版本
    for v in range(5):
        model_path = f"./checkpoint/model{v}"
        results = evaluate_model(model_path, test_data, text_build_model[v])

        print(f"\nResults for model {text_build_model[v]}:")
        print(f"Micro-Precision: {results['precision']:.4f}")
        print(f"Micro-Recall: {results['recall']:.4f}")
        print(f"Micro-F1: {results['f1']:.4f}")


if __name__ == "__main__":
    main()

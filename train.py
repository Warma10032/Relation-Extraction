import torch
import csv
import json
from datetime import datetime

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from transformers import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

from data import *
from log import *

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


MAX_LEN = 300  # 设置tokenizer的最大长度
logger = setup_logger(
    f'./output/log/results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)
version = 5
text_build_model = ["basic1", "basic2", "QA", "entity_marked1", "entity_marked2"]
model_name = r"hfl/chinese-roberta-wwm-ext"


# 读取jsonl文件
def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


train_data = read_jsonl("./dataset/train.jsonl")
val_data = read_jsonl("./dataset/val.jsonl")
test_data = read_jsonl("./dataset/test.jsonl")


for v in range(version):
    # 加载预训练的BERT分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = RelationshipDataset(
        train_data, tokenizer, MAX_LEN, text_build_model[v]
    )
    val_dataset = RelationshipDataset(val_data, tokenizer, MAX_LEN, text_build_model[v])
    test_dataset = RelationshipDataset(
        test_data, tokenizer, MAX_LEN, text_build_model[v]
    )

    # 获取数据集的标签列表
    labels = [train_dataset.label_map[label] for label in train_dataset.labels]

    train_dataloader_b = DataLoader(
        train_dataset,
        batch_sampler=BalancedBatchSampler(train_dataset, labels, batch_size=20),
    )
    train_dataloader_p = DataLoader(
        train_dataset,
        batch_sampler=PiorityBatchSampler(train_dataset, labels, batch_size=16),
    )

    val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    # 加载预训练的中文 BERT 模型
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=10)
    model.resize_token_embeddings(len(tokenizer))

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

    class_weights = torch.ones(10, device=device)
    # 增加关注的类别权重
    class_weights[4] = 2.0  # 并发症
    class_weights[8] = 2.0  # 相关（导致）

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # 训练过程
    epochs = 4
    losses = []
    for epoch in range(int(epochs / 2)):
        model.train()
        for batch in train_dataloader_b:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            losses.append(loss.item())

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        scheduler.step()

        # 模型评估
        model.eval()
        predictions = []
        true_labels = []

        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

        # 输出结果
        logger.info(f"\n{'='*50}")
        logger.info(
            f"Model {text_build_model[v]}. Balanced epoch {epoch+1} training done"
        )
        logger.info(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
        logger.info(f"\n{classification_report(true_labels, predictions, digits=4)}")

    for epoch in range(int(epochs / 2)):
        model.train()
        for batch in train_dataloader_p:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            losses.append(loss.item())

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        scheduler.step()

        # 模型评估
        model.eval()
        predictions = []
        true_labels = []

        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

        # 输出结果
        logger.info(f"\n{'='*50}")
        logger.info(
            f"Model {text_build_model[v]}. Piority epoch {epoch+1} training done"
        )
        logger.info(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
        logger.info(f"\n{classification_report(true_labels, predictions, digits=4)}")

    # 将损失值保存到文件中（例如 CSV 文件）
    with open(f"./output/loss/train_losses{v}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "loss"])  # 写入表头
        for i, loss in enumerate(losses):
            writer.writerow([i + 1, f"{loss:.6f}"])  # 保留6位小数写入

    # 确保所有模型参数是连续的
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    model.save_pretrained(f"./checkpoint/model{v}")
    tokenizer.save_pretrained(f"./checkpoint/model{v}")

from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import torch


class RelationshipDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, text_build_mode="basic"):
        self.sentences = [sample["sentence"] for sample in data]
        self.h_entities = [sample["h"] for sample in data]
        self.t_entities = [sample["t"] for sample in data]
        self.labels = [sample["r"] for sample in data]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {
            "临床表现": 0,
            "药物治疗": 1,
            "同义词": 2,
            "病因": 3,
            "并发症": 4,
            "病理分型": 5,
            "实验室检查": 6,
            "辅助治疗": 7,
            "相关（导致）": 8,
            "影像学检查": 9,
        }

        # 将特殊标记添加到tokenizer中
        if text_build_mode == "entity_marked1":
            # 先检查这些标记是否已经存在
            new_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
            tokens_to_add = []
            for token in new_tokens:
                if token not in self.tokenizer.get_vocab():
                    tokens_to_add.append(token)

            if tokens_to_add:
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": tokens_to_add}
                )
        elif text_build_mode == "entity_marked2":
            new_tokens = ["[实体1]", "[/实体1]", "[实体2]", "[/实体2]"]
            tokens_to_add = []
            for token in new_tokens:
                if token not in self.tokenizer.get_vocab():
                    tokens_to_add.append(token)

            if tokens_to_add:
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": tokens_to_add}
                )

        self.text_build_mode = text_build_mode

    def build_text(self, sentence, h, t):
        if self.text_build_mode == "basic1":
            # 基础模式
            return f"[CLS] {h} [SEP] {sentence} [SEP] {t}"
        elif self.text_build_mode == "basic2":
            # 基础模式2
            return f"[CLS] {h} [SEP] {t} [SEP] {sentence}"
        elif self.text_build_mode == "QA":
            # 问答模板模式
            return f"[CLS] {h}和{t}之间的关系是什么？[SEP] {sentence}"
        elif self.text_build_mode == "entity_marked1":
            # 实体标记模式1
            marked_sentence = sentence.replace(h, f"[E1]{h}[/E1]").replace(
                t, f"[E2]{t}[/E2]"
            )
            return f"[CLS] {marked_sentence} [SEP]"
        elif self.text_build_mode == "entity_marked2":
            # 实体标记模式2
            marked_sentence = sentence.replace(h, f"[实体1]{h}[/实体1]").replace(
                t, f"[实体2]{t}[/实体2]"
            )
            return f"[CLS] {marked_sentence} [SEP]"

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]
        h = self.h_entities[item]
        t = self.t_entities[item]
        label = self.label_map[self.labels[item]]  # 将关系标签映射为数字

        # 构建文本
        text = self.build_text(sentence, h, t)

        # 使用BERT分词器对文本进行编码
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].flatten()
        attention_mask = encoded["attention_mask"].flatten()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, labels, batch_size):
        self.labels = labels
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = len(set(labels))
        self.samples_per_class = batch_size // self.num_classes

        # 按类别索引数据
        self.label_to_indices = {}
        for i in range(self.num_classes):
            self.label_to_indices[i] = np.where(np.array(labels) == i)[0]

        # 计算每个类别可以生成的完整batch数
        self.n_batches = (
            min([len(indices) for indices in self.label_to_indices.values()])
            // self.samples_per_class
        )

    def __iter__(self):
        # 为每个类别创建索引副本并打乱
        used_indices = {}
        available_indices = {}
        for label in range(self.num_classes):
            available_indices[label] = self.label_to_indices[label].copy()
            np.random.shuffle(available_indices[label])
            used_indices[label] = 0

        # 生成指定数量的batch后停止
        for _ in range(self.n_batches):
            indices = []
            for class_id in range(self.num_classes):
                class_indices = available_indices[class_id]
                indices.extend(
                    class_indices[
                        used_indices[class_id] : used_indices[class_id]
                        + self.samples_per_class
                    ]
                )
                used_indices[class_id] += self.samples_per_class

            yield indices

    def __len__(self):
        return self.n_batches


class PiorityBatchSampler(Sampler):
    def __init__(self, dataset, labels, batch_size=16):
        self.labels = labels
        self.dataset = dataset
        self.batch_size = batch_size  # 固定为16
        self.num_classes = len(set(labels))

        # 特殊类别（第4类和第8类）每个batch的样本数
        self.special_classes = [4, 8]  # 索引为4和8的类别
        self.samples_per_special = 4  # 每个特殊类别4个样本

        # 其他类别每个batch的样本数
        self.regular_classes = [
            i for i in range(self.num_classes) if i not in self.special_classes
        ]
        self.samples_per_regular = 1  # 其他类别各1个样本

        # 按类别索引数据
        self.label_to_indices = {}
        for i in range(self.num_classes):
            self.label_to_indices[i] = np.where(np.array(labels) == i)[0]

        special_batches = min(
            [
                len(self.label_to_indices[i]) // self.samples_per_special
                for i in self.special_classes
            ]
        )
        regular_batches = min(
            [
                len(self.label_to_indices[i]) // self.samples_per_regular
                for i in self.regular_classes
            ]
        )
        self.n_batches = min(special_batches, regular_batches)

    def __iter__(self):
        # 为每个类别创建索引副本并打乱
        available_indices = {}
        used_indices = {}

        for label in range(self.num_classes):
            available_indices[label] = self.label_to_indices[label].copy()
            np.random.shuffle(available_indices[label])
            used_indices[label] = 0

        # 生成指定数量的batch
        for _ in range(self.n_batches):
            indices = []

            # 添加特殊类别的样本（4和8类各4个）
            for class_id in self.special_classes:
                class_indices = available_indices[class_id]
                indices.extend(
                    class_indices[
                        used_indices[class_id] : used_indices[class_id]
                        + self.samples_per_special
                    ]
                )
                used_indices[class_id] += self.samples_per_special

            # 添加其他类别的样本（每类1个）
            for class_id in self.regular_classes:
                class_indices = available_indices[class_id]
                indices.extend(
                    class_indices[
                        used_indices[class_id] : used_indices[class_id]
                        + self.samples_per_regular
                    ]
                )
                used_indices[class_id] += self.samples_per_regular

            # 打乱当前batch中的样本顺序
            np.random.shuffle(indices)
            yield indices

    def __len__(self):
        return self.n_batches

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast, Trainer, TrainingArguments


def finetune(train_path, val_path, test_path):
    def read_wnut(file_path):
        file_path = Path(file_path)

        raw_text = file_path.read_text().strip()
        raw_docs = re.split(r'\n\t?\n', raw_text)
        token_docs = []
        tag_docs = []
        for doc in raw_docs:
            tokens = []
            tags = []
            for line in doc.split('\n'):
                token, tag = line.split('\t')
                tokens.append(token)
                tags.append(tag)
            token_docs.append(tokens)
            tag_docs.append(tags)

        return token_docs, tag_docs

    train_texts, train_tags = read_wnut(train_path)
    val_texts, val_tags = read_wnut(val_path)
    test_texts, test_tags = read_wnut(test_path)

    # Create encodings for tags
    unique_tags = set(tag for doc in train_tags for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    # Create encodings for tokens
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                                truncation=True)
    val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                              truncation=True)
    test_encodings = tokenizer(test_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                               truncation=True)

    def encode_tags(tags, encodings):
        labels = [[tag2id[tag] for tag in doc] for doc in tags]
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            # Create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
            arr_offset = np.array(doc_offset)

            # Set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

        return encoded_labels

    train_labels = encode_tags(train_tags, train_encodings)
    val_labels = encode_tags(val_tags, val_encodings)
    test_labels = encode_tags(test_tags, test_encodings)

    class WNUTDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_encodings.pop('offset_mapping')  # We don't want to pass this to the model
    val_encodings.pop('offset_mapping')
    test_encodings.pop('offset_mapping')

    train_dataset = WNUTDataset(train_encodings, train_labels)
    val_dataset = WNUTDataset(val_encodings, val_labels)
    test_dataset = WNUTDataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        output_dir='./results',  # Output directory
        num_train_epochs=3,  # Total number of training epochs
        per_device_train_batch_size=16,  # Batch size per device during training
        per_device_eval_batch_size=64,  # Batch size for evaluation
        warmup_steps=500,  # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # Strength of weight decay
        logging_dir='./logs'  # Directory for storing logs
    )

    model = DistilBertForTokenClassification.from_pretrained(
        'models/distilbert-base-cased',
        num_labels=len(unique_tags)
    )

    def align_predictions(predictions, label_ids):
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(id2tag[label_ids[i][j]])
                    preds_list[i].append(id2tag[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(p):
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        return {
            'accuracy_score': accuracy_score(out_label_list, preds_list),
            'precision': precision_score(out_label_list, preds_list),
            'recall': recall_score(out_label_list, preds_list),
            'f1': f1_score(out_label_list, preds_list),
        }

    trainer = Trainer(
        model=model,  # The instantiated Transformers model to be trained
        args=training_args,  # Training arguments, defined above
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,  # Training dataset
        eval_dataset=val_dataset  # Evaluation dataset
    )

    trainer.train()
    trainer.evaluate()

    # Predict
    predictions, label_ids, metrics = trainer.predict(test_dataset)
    preds_list, _ = align_predictions(predictions, label_ids)

    with open('results/test_results.txt', 'w') as writer:
        for key, value in metrics.items():
            writer.write(f'{key} = {value}\n')

    with open('results/test_predictions.txt', 'w') as writer:
        with open(test_path, 'r') as f:
            example_id = 0
            for line in f:
                if line.startswith('-DOCSTART') or line == '' or line == '\n':
                    writer.write(line)
                    if not preds_list[example_id]:
                        example_id += 1
                elif preds_list[example_id]:
                    writer.write(line.split()[0] + ' ' + preds_list[example_id].pop(0) + '\n')
                else:
                    print(f'Maximum sequence length exceeded: No prediction for {line.split()[0]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path')
    parser.add_argument('val_path')
    parser.add_argument('test_path')

    args = parser.parse_args()

    finetune(args.train_path, args.val_path, args.test_path)

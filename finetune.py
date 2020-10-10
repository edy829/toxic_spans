import argparse
import re
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast, Trainer, TrainingArguments


def finetune(file_path):
    # TODO: What if toxic word also appears in a non-toxic context?
    df = pd.read_csv(file_path)
    df['spans'] = df.spans.apply(literal_eval)

    texts = []
    tags = []

    for _, row in df.iterrows():
        # Remove intermediary span indices, e.g., [15, 16, 17, 18, 19, 27, 28, 29, 30, 31] -> [15, 19, 27, 31]
        spans = []
        for i in range(len(row['spans'])):
            if i == 0 or i == len(row['spans']) - 1:
                spans.append(row['spans'][i])
            elif row['spans'][i] - row['spans'][i - 1] > 1:
                spans.extend([row['spans'][i - 1], row['spans'][i]])

        # Identify all toxic words in row
        toxic = []
        for i in range(0, len(spans), 2):
            toxic.append(row['text'][spans[i]:spans[i + 1] + 1])

        sen = []
        sen_tags = []

        # Split row into words and store each word and its corresponding tag
        for word in re.findall(r"[\w']+|[.,!?;]", row['text']):
            sen.append(word)
            if word in toxic:
                sen_tags.append('B-toxic') if sen_tags and sen_tags[-1] == 'O' else sen_tags.append('I-toxic')
            else:
                sen_tags.append('O')

        texts.append(sen)
        tags.append(sen_tags)

    train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2)

    # Create encodings for tags
    unique_tags = set(tag for doc in tags for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    # Create encodings for tokens
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                                truncation=True)
    val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
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

    train_encodings.pop("offset_mapping")  # We don't want to pass this to the model
    val_encodings.pop("offset_mapping")
    train_dataset = WNUTDataset(train_encodings, train_labels)
    val_dataset = WNUTDataset(val_encodings, val_labels)

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

    trainer = Trainer(
        model=model,  # The instantiated Transformers model to be trained
        args=training_args,  # Training arguments, defined above
        train_dataset=train_dataset,  # Training dataset
        eval_dataset=val_dataset  # Evaluation dataset
    )

    trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path')
    finetune(parser.parse_args().file_path)

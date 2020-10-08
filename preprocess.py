import argparse
import re
from ast import literal_eval

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(file_path):
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
        toxic_words = []
        for i in range(0, len(spans), 2):
            toxic_words.append(row['text'][spans[i]:spans[i + 1] + 1])

        # Split row into words and store each word with its corresponding tag
        for word in re.findall(r"[\w']+|[.,!?;]", row['text']):
            texts.append(word)
            # TODO: B- I- in tags
            tags.append('B-toxic') if word in toxic_words else tags.append('O')

    train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path')
    preprocess_data(parser.parse_args().file_path)

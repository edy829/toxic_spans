import argparse
import os
import re
from ast import literal_eval

import pandas as pd


def preprocess(file_path):
    df = pd.read_csv(file_path)
    df['spans'] = df.spans.apply(literal_eval)

    texts = []
    tags = []

    for _, row in df.iterrows():
        spans = row['spans']
        text = re.findall(r"\w+(?:'\w+)*|[^\w]", row['text'].replace('\n', ' '))

        sentence = []
        sentence_tags = []

        offset = 0
        for word in text:
            length = len(word)
            if word.isspace():
                offset += length
                continue

            sentence.append(word)

            toxic = False
            for i in range(length):
                if i + offset in spans:
                    toxic = True

            if toxic:
                try:
                    if sentence_tags[-1] == 'O':
                        sentence_tags.append('B-toxic')
                    else:
                        sentence_tags.append('I-toxic')
                except IndexError:
                    sentence_tags.append('B-toxic')
            else:
                sentence_tags.append('O')

            offset += length

        texts.append(sentence)
        tags.append(sentence_tags)

    with open(
            f'{os.path.dirname(file_path)}/preprocessed/{os.path.splitext(os.path.basename(file_path))[0]}.txt',
            'w'
    ) as writer:
        for sentence, sentence_tags in zip(texts, tags):
            for text, tag in zip(sentence, sentence_tags):
                writer.write(f'{text}\t{tag}\n')
            writer.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path')
    preprocess(parser.parse_args().file_path)

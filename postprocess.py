import argparse
import csv
import re
from ast import literal_eval

import pandas as pd


def postprocess(result_path, test_path):
    # Read results
    with open(result_path, 'r') as f:
        lines = f.readlines()

    texts = []
    tags = []

    sen = []
    sen_tags = []

    for i, line in enumerate(lines):
        if line == '\n':
            texts.append(sen)
            tags.append(sen_tags)

            sen = []
            sen_tags = []

            continue

        text, tag = re.split(r'\t+', line)
        sen.append(text)
        sen_tags.append(tag.rsplit()[0])

    texts.append(sen)
    tags.append(sen_tags)

    # Read test data
    df = pd.read_csv(test_path)
    df['spans'] = df.spans.apply(literal_eval)

    original_texts = []

    for _, row in df.iterrows():
        original_text = re.findall(r"\w+(?:'\w+)*|[^\w]", row['text'].replace('\n', ' '))
        original_texts.append(original_text)

    with open('results/test_predictions.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['spans', 'text'])

    # Convert tags to spans
    for sen, sen_tags in zip(texts, tags):
        spans = []

        # Find original text
        original = []
        for original_text in original_texts:
            if sen == [text for text in original_text if not text.isspace()]:
                original = original_text
                break

        # Calculate spans
        char_offset = 0
        list_offset = 0
        for i in range(len(original)):
            word = original[i]
            length = len(word)
            if word.isspace():
                char_offset += length
                list_offset += 1
                continue

            if sen_tags[i - list_offset] == 'B-toxic' or sen_tags[i - list_offset] == 'I-toxic':
                spans.extend(list(range(char_offset, char_offset + length)))

            char_offset += length

        with open('results/test_predictions.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([spans, ''.join(original)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_path')
    parser.add_argument('test_path')

    args = parser.parse_args()

    postprocess(args.result_path, args.test_path)

import string
import os
import json

VOCAB_PATH = './data/imdb.vocab'

def is_punc(char):
    return char in string.punctuation

def lowercase_sentence(comment: str) -> str:
    return comment.lower()


def separate_punctuation(comment: str) -> str:
    res = []
    for word in comment.split():
        start = 0 # start index of current separation
        for i, c in enumerate(word):
            if is_punc(c):
                if start < i:
                    res.append(word[start: i])
                res.append(c)
                start = i + 1
        if start < len(word):
            res.append(word[start:])
    return ' '.join(res)


def preprocess_file(file_path: str) -> str:
    with open(file_path) as file:
        comment = ' '.join([line.strip() for line in file])
        comment = lowercase_sentence(comment)
        comment = separate_punctuation(comment)
        return comment


def load_vocab():
    vocabs = set()
    with open(VOCAB_PATH) as vocab_file:
        for word in vocab_file:
            vocabs.add(word.strip())
    return vocabs


def build_bag_of_word_vector(comment: str, vocabs: set):
    vector = {}
    for word in comment.split():
        if word in vocabs:
            if word in vector:
                vector[word] += 1
            else:
                vector[word] = 1
    return vector


def save_file(file_content, file_path):
    with open(file_path, 'w') as file:
        json.dump(file_content, file)


def preprocess_folder(folder_path: str, output_folder: str):
    vocabs = load_vocab()

    for filename in os.listdir(folder_path):
        comment = preprocess_file(f'{folder_path}/{filename}')
        vector = build_bag_of_word_vector(comment, vocabs)
        save_file(vector, f'{output_folder}/{filename.split(".")[0]}.json')


def testing():
    output_folder = './preprocessed/neg'
    filename = '10_2.txt'
    vocabs = load_vocab()
    comment = preprocess_file(f'./data/train/neg/{filename}')
    print(comment)
    vector = build_bag_of_word_vector(comment, vocabs)
    print(vector)

    save_file(vector, f'{output_folder}/{filename.split(".")[0]}.json')


# testing()
preprocess_folder('./data/train/neg', './preprocessed/neg')
preprocess_folder('./data/train/pos', './preprocessed/pos')

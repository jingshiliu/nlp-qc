import string
import os
import json


# --------------------------------------------------- load file ------------------------------------------------------ #

def load_vocab(vocab_path):
    vocabs = set()
    with open(vocab_path) as vocab_file:
        for word in vocab_file:
            vocabs.add(word.strip())
    return vocabs


def save_file(file_content, file_path):
    with open(file_path, 'w') as file:
        file.write(file_content)


# ------------------------------------------------ preprocess files -------------------------------------------------- #

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


def preprocess_comment(comment: str) -> str:
    return separate_punctuation(lowercase_sentence(comment))


def preprocess_file(file_path: str) -> str:
    with open(file_path) as file:
        comment = ' '.join([line.strip() for line in file])
        comment = lowercase_sentence(comment)
        comment = separate_punctuation(comment)
        return comment


def build_bag_of_word_vector(comment: str, vocabs: set):
    vector = {}
    for word in comment.split():
        if word in vocabs:
            if word in vector:
                vector[word] += 1
            else:
                vector[word] = 1
    return vector


def preprocess_folder(folder_path: str, vocabs):
    vector_list = []
    for filename in os.listdir(folder_path):
        comment = preprocess_file(f'{folder_path}/{filename}')
        vector_list.append(build_bag_of_word_vector(comment, vocabs))
    return vector_list


def preprocess(folder_path1, folder_path2, vocab_path, path1_class, path2_class, output_path):
    # label######{json format of vector}
    # ###### is the separator of column to access easier
    vocabs = load_vocab(vocab_path)
    folder_path1_vectors = preprocess_folder(folder_path1, vocabs)
    folder_path2_vectors = preprocess_folder(folder_path2, vocabs)

    res = []
    for vector in folder_path1_vectors:
        res.append(f'{path1_class}#####{json.dumps(vector)}')
    for vector in folder_path2_vectors:
        res.append(f'{path2_class}#####{json.dumps(vector)}')
    save_file('\n'.join(res), output_path)


# ---------------------------------------------------- testing  ----------------------------------------------------- #


def testing():

    preprocess(folder_path1='./data/train/neg',
               folder_path2='./data/train/pos',
               vocab_path="./data/imdb.vocab",
               path1_class='neg',
               path2_class='pos',
               output_path='./preprocessed/train_BOW.txt'
               )


# preprocess_folder('./data/train/neg', './preprocessed/neg')
# preprocess_folder('./data/train/pos', './preprocessed/pos')

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


def contains_strong_pos_word(words:list) -> bool:
    positive_words = {'excellent', 'amazing', 'great', 'fantastic', 'outstanding', 'terrific', 'phenomenal', 'superb',
                      'brilliant', 'impressive'}
    for word in words:
        if word in positive_words:
            return True
    return False


def contains_strong_neg_word(words: list) -> bool:
    negative_words = {'disappointing', 'terrible', 'awful', 'horrible', 'dreadful', 'abysmal', 'appalling', 'atrocious',
                      'repulsive', 'disgusting'}
    for word in words:
        if word in negative_words:
            return True
    return False


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
        res.append(f'{path1_class}#####{json.dumps(vector)}#####{int(contains_strong_pos_word(list(vector.keys())))}#####{int(contains_strong_neg_word(list(vector.keys())))}')
    for vector in folder_path2_vectors:
        res.append(f'{path2_class}#####{json.dumps(vector)}#####{int(contains_strong_pos_word(list(vector.keys())))}#####{int(contains_strong_neg_word(list(vector.keys())))}')
    save_file('\n'.join(res), output_path)



import os
import json

VOCAB_PATH = './data/imdb.vocab'

# --------------------------------------------------- load file ------------------------------------------------------ #

def load_vocab() -> set:
    vocabs = set()
    with open(VOCAB_PATH) as vocab_file:
        for word in vocab_file:
            vocabs.add(word.strip())
    return vocabs


def save_file(file_content, file_path):
    with open(file_path, 'w') as file:
        json.dump(file_content, file)


def load_json(file_path: str) -> dict:
    with open(file_path) as file:
        return json.load(file)

def load_one_vector(vector_file_path: str) -> dict:
    return load_json(vector_file_path)


def load_naive_bayes_classifier(classifier_path: str) -> dict:
    return load_json(classifier_path)


# ------------------------------------------ train naive bayes classifier -------------------------------------------- #

def initialize_counter() -> dict:
    '''
    load vocab and build  a dictionary that contains all vocab as key, and value set to 0
    :return: a dictionary that contains all vocab as key, and value set to 0
    '''
    vocabs = load_vocab()
    return {vocab: 0 for vocab in vocabs}


def aggregate_vectors_into_counter(counter: dict, vector_folder_path: str):
    for vector_filename in os.listdir(vector_folder_path):
        vector = load_one_vector(f'{vector_folder_path}/{vector_filename}')
        for word, freq in vector.items():
            counter[word] += freq


def train_naive_bayes_class_recognizer(counter: dict, total_token: int) -> dict:
    recognizer = {}
    total_vocab = len(counter)
    add_one_smoothing_total_token = total_vocab + total_token
    for word, freq in counter.items():
        recognizer[word] = (freq + 1) / add_one_smoothing_total_token
    return recognizer


def naive_bayes_class_recognizer(vector_folder_path):
    counter = initialize_counter()
    aggregate_vectors_into_counter(counter, vector_folder_path)
    total_token = sum(counter.values())
    class_recognizer = train_naive_bayes_class_recognizer(counter, total_token)

    return class_recognizer, total_token


def naive_bayes(pos_folder_path, neg_folder_path, result_model_path):
    neg_recognizer, neg_total_token = naive_bayes_class_recognizer('./preprocessed/neg')
    pos_recognizer, pos_total_token = naive_bayes_class_recognizer('./preprocessed/pos')

    neg_prior_prob = neg_total_token / (neg_total_token + pos_total_token)
    pos_prior_prob = pos_total_token / (neg_total_token + pos_total_token)

    naive_bayes_classifier = {"neg": neg_recognizer,
                              "pos": pos_recognizer,
                              "neg_prior": neg_prior_prob,
                              "pos_prior": pos_prior_prob
                              }

    save_file(naive_bayes_classifier, result_model_path)
    return naive_bayes_classifier


# ----------------------------------------------- evaluate test data ------------------------------------------------ #

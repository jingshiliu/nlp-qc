import os
import json

from pre_process import preprocess_comment, preprocess_folder

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


def naive_bayes(class_1_folder_path, class_2_folder_path, result_model_path, class_1="pos", class_2="neg"):
    class_1_recognizer, class_1_total_token = naive_bayes_class_recognizer(class_1_folder_path)
    class_2_recognizer, class_2_total_token = naive_bayes_class_recognizer(class_2_folder_path)

    class_1_prior_prob = class_1_total_token / (class_1_total_token + class_2_total_token)
    class_2_prior_prob = class_2_total_token / (class_1_total_token + class_2_total_token)

    naive_bayes_classifier = {class_1: class_1_recognizer,
                              class_2: class_2_recognizer,
                              f'{class_1}_prior': class_1_prior_prob,
                              f'{class_2}_prior': class_2_prior_prob,
                              "class_1": class_1,
                              "class_2": class_2
                              }

    save_file(naive_bayes_classifier, result_model_path)
    return naive_bayes_classifier


# ----------------------------------------------- evaluate test data ------------------------------------------------ #

def compute_prob(comment: str | list, class_recognizer: dict, prior_prob: float) -> float:
    if type(comment) is str:
        comment = comment.split()

    prob = prior_prob
    for word in comment:
        if word not in class_recognizer:
            continue
        prob *= class_recognizer[word]
    return prob


class NaiveBayesClassifier:
    def __init__(self, path_to_model="", model=None):
        self.model = model
        self.class_1 = None
        self.class_2 = None
        if not model and path_to_model:
            self.load_model(path_to_model)

    def load_model(self, path_to_model: str):
        with open(path_to_model) as model_file:
            self.model = json.load(model_file)
            self.class_1 = self.model["class_1"]
            self.class_2 = self.model["class_2"]

    def classify(self, comment: str):
        comment = preprocess_comment(comment)
        word_list = comment.split()
        class_1_prob = compute_prob(word_list, self.model[self.class_1], self.model[f"{self.class_1}_prior"])
        class_2_prob = compute_prob(word_list, self.model['neg'], self.model[f"{self.class_2}_prior"])

        return self.class_1 if class_1_prob > class_2_prob else self.class_2


# ------------------------------------------ main (training and evaluate) ------------------------------------------- #


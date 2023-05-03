import math

from utils import *
from collections import defaultdict


def train_bigram(preprocessed_data_path, model_save_path, skip=0):
    """
    train bigram language model, the start sentence padding <s> is ignored
    :param skip: use to skip first token, mainly for  <s>
    :param preprocessed_data_path:
    :param model_save_path:
    :return:
    """
    with open(preprocessed_data_path) as training_corpus:
        training_data = file_to_list_list(training_corpus)
        token_counter = count_token(training_data)
        bigram_token_counter = ngram_count_token(training_data, skip=skip)

        bigram_prob = {}
        for token_seq in bigram_token_counter.keys():
            bigram_prob[f'{token_seq[0]} {token_seq[1]}'] = bigram_token_counter[token_seq] / token_counter[
                token_seq[0]]
            # bigram_prob[f'{token_seq[0]} {token_seq[1]}'] = f'{bigram_token_counter[token_seq]} / {token_counter[token_seq[0]]}'

        save_language_model(bigram_prob, model_save_path)


def bigram_evaluate_log_prob(data_list_list: list[list], bigram_model_path):
    bigram_model = load_language_model(bigram_model_path)
    prob = 0
    for sentence in data_list_list:
        for i in range(1, len(sentence)):
            bigram_token = f'{sentence[i - 1]} {sentence[i]}'
            try:
                # print(bigram_token, bigram_model.get(bigram_token, 0), math.log2(bigram_model.get(bigram_token, 0)))
                prob += math.log2(bigram_model.get(bigram_token, 0))
            except:
                # print(bigram_token, 0, 0)
                prob += 0

    return prob


def bigram_evaluate_log_prob_add_one_smoothing(data_list_list: list[list], training_data_path: str, skip=0):
    training_data = None
    with open(training_data_path) as training:
        training_data = file_to_list_list(training)
    bigram_token_counter = ngram_count_token(training_data, skip=skip)
    unigram_token_counter = count_token(training_data)
    del unigram_token_counter['<s>']

    total_token_types = len(unigram_token_counter.keys())
    total_bigram_token_types = total_token_types ** 2
    # print(total_token_types, total_bigram_token_types)
    prob = 0
    for sentence in data_list_list:
        for i in range(1, len(sentence)):
            bigram_token = (sentence[i - 1], sentence[i])

            token_prob = (bigram_token_counter.get(bigram_token, 0) + 1) \
                         / \
                         (unigram_token_counter.get(sentence[i - 1], 0) + total_bigram_token_types)
            # print(bigram_token_counter.get(bigram_token, 0) + 1)
            # print((unigram_token_counter.get(sentence[i - 1], 0) + total_bigram_token_types))
            # print(bigram_token, token_prob, math.log2(token_prob))
            prob += math.log2(token_prob)

    return prob


train_bigram('./Data/pre_processed_training_data.txt', './Models/bigram_model1.json', skip=1)

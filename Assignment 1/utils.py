import collections
import json
from collections import defaultdict


def file_to_list_list(corpus):
    return [sentence.split(' ') for sentence in corpus.read().split('\n')]


def count_token(data: list[list], subsequence=1, skip=0) -> collections.defaultdict:
    token_counter = defaultdict(int)
    if subsequence == 1:
        for sentence in data:
            for i in range(skip, len(sentence)):
                token = sentence[i]
                token_counter[token] = token_counter.get(token, 0) + 1
    else:
        return ngram_count_token(data, subsequence)
    return token_counter


def ngram_count_token(data: list[list], ngram=2, skip=0) -> collections.defaultdict:
    """
    You should always do defaultdict.get(key, 0) for all ngram tuple count retrieval because the returned defaultdict
    does not contain the ngram parameters that have 0 count for saving memory purpose
    :param skip: number of token to be skipped in each sentence, mainly used to ignore <s>
    :param data:
    :param ngram: 2 for bigram, 3 for trigram
    :return: a defaultdict counter
    """
    token_counter = defaultdict(int)
    for sentence in data:
        for i in range(ngram - 1 + skip, len(sentence)):

            token_seq = tuple([sentence[j] for j in range(i - ngram + 1, i + 1)])
            token_counter[token_seq] = token_counter.get(token_seq, 0) + 1
    return token_counter


def load_language_model(model_path):
    with open(model_path) as model:
        return defaultdict(int, json.load(model))


def save_language_model(model, model_path):
    with open(model_path, 'w') as model_file:
        json.dump(model, model_file)


def data_list_to_str(data: list[list]):
    return '\n'.join([' '.join(sentence) for sentence in data])


def map_test_not_in_training_unk(test_path, training_path):
    with open(test_path, 'r') as test_file:
        with open(training_path) as training_file:
            training_token_counter = count_token(file_to_list_list(training_file))
            test_data_list = file_to_list_list(test_file)

            for sentence in test_data_list:
                for i in range(len(sentence)):
                    if sentence[i] not in training_token_counter:
                        sentence[i] = '<unk>'

            with open('./Data/test_token_not_in_training_to_unk.txt', 'w') as test_target_file:
                test_target_file.write(data_list_to_str(test_data_list))


def calculate_ppl(log_prob, total_token):
    return 2 ** (-1 * (log_prob / total_token))

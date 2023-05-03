import math

from utils import *


def train_unigram(preprocessed_data_path, model_save_path, skip=0):
    with open(preprocessed_data_path) as training_corpus:
        training_data = file_to_list_list(training_corpus)
        token_counter = count_token(training_data, skip=skip)
        token_total = sum(token_counter.values())

        # unigram_prob = {token: f'{token_counter[token]} / {token_total}' for token in token_counter.keys()}
        unigram_prob = {token: token_counter[token] / token_total for token in token_counter.keys()}

        save_language_model(unigram_prob, model_save_path)


def unigram_evaluate_log_prob(data_list_list: list[list], unigram_model_path):
    unigram_model = load_language_model(unigram_model_path)
    prob = 0
    for sentence in data_list_list:
        for token in sentence:
            print(token)
            print(unigram_model.get(token, 0))
            print(math.log2(unigram_model.get(token, 0)))
            prob += math.log2(unigram_model.get(token, 0))

    return prob


# data_list_list = [
#     ['a', 'unigram', 'maximum', 'likelihood', 'model', '.', '</s>'],
# ['a', 'bigram', 'maximum', 'likelihood', 'model', '.', '</s>'],
# ['a', 'bigram', 'model', 'with', 'add-one', 'smoothing', '.', '</s>']
# ]

# train_unigram('./Test/test_training.txt', './Test/test_model.json')
train_unigram('./Data/pre_processed_training_data.txt', './Models/unigram_model1.json')

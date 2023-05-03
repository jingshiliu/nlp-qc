from utils import *


def lower_and_pad(file_data):
    """
    :param file_data: a file to be made lower case and each sentence to padded
    :return: tuple(list[][], dict{})
            each list[i] is a sentence, and list[i][j] is a word
            dict{} a counter dictionary
    """
    processed_data = []
    word_count = {}

    line = file_data.readline().split()
    while line:
        processing_line = ['<s>']
        for token in line:
            token = token.lower()
            if token in word_count:
                word_count[token] += 1
            else:
                word_count[token] = 1
            processing_line.append(token)
        processing_line.append('</s>')
        processed_data.append(processing_line)
        line = file_data.readline().split()

    return processed_data, word_count


def get_words_appeared_once_only(word_count: dict) -> set:
    words_appeared_once = set()
    for word in word_count.keys():
        if word_count[word] == 1:
            words_appeared_once.add(word)
    # fancy one liner
    # words_appeared_once = set(dict(filter(lambda x: x[1] == 1, word_count.items())).keys())
    return words_appeared_once


def replace_words_appeared_once_with_unk(data: list[list], words_appeared_once: set):
    for sentence in data:
        for i in range(len(sentence)):
            if sentence[i] in words_appeared_once:
                sentence[i] = '<unk>'


def write_data_to_file(data: str, file_obj=None, file_path=None):
    if not file_obj:
        if not file_path:
            raise Exception('No path and file object provided')
        file_obj = open(file_path, 'w')

    file_obj.write(data)
    file_obj.close()


def pre_process_training_data(training_data_path: str, pre_processed_data_path: str):
    with open(training_data_path, 'r') as training_corpus:
        with open(pre_processed_data_path, 'w') as pre_processed_training_corpus:
            processing_data, word_count = lower_and_pad(training_corpus)
            words_appeared_once = get_words_appeared_once_only(word_count)
            replace_words_appeared_once_with_unk(processing_data, words_appeared_once)
            pre_processed_training_corpus.write(data_list_to_str(processing_data))


def lower_and_pad_file(data_path: str, save_path: str):
    with open(data_path) as file:
        with open(save_path, 'w') as output_file:
            lower_and_padded_data, word_count = lower_and_pad(file)
            output_file.write(data_list_to_str(lower_and_padded_data))


def pre_process_test_data(data_path: str, save_path: str, pre_processed_training_path):
    with open(data_path) as file:
        with open(save_path, 'w') as output_file:
            with open(pre_processed_training_path) as training_file:
                training_token_counter = count_token(file_to_list_list(training_file))
                lower_and_padded_data, word_count = lower_and_pad(file)

                for sentence in lower_and_padded_data:
                    for i in range(len(sentence)):
                        if sentence[i] not in training_token_counter:
                            sentence[i] = '<unk>'

                output_file.write(data_list_to_str(lower_and_padded_data))


# pre_process_test_data('./Data/test.txt', './1.txt', './Data/pre_processed_training_data.txt')
# pre_process_data('./Data/train-Spring2023.txt', './Data/pre_processed_training_data.txt')
# pre_process_data('./Data/test.txt', './Data/pre_processed_test_corpus.txt')

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


def data_list_to_str(data: list[list]):
    return '\n'.join([' '.join(sentence) for sentence in data])


def pre_process_data(training_data_path: str, pre_processed_data_path: str):
    with open(training_data_path, 'r') as training_data:
        with open(pre_processed_data_path, 'w') as pre_processed_data:
            processing_data, word_count = lower_and_pad(training_data)
            words_appeared_once = get_words_appeared_once_only(word_count)
            replace_words_appeared_once_with_unk(processing_data, words_appeared_once)
            pre_processed_data.write(data_list_to_str(processing_data))

            # +2 bc not counted </s> and <unk>
            print('Unique Words:', len(word_count) + 2)


pre_process_data('Data/train-Spring2023.txt', 'Data/pre_processed_training_data.txt')

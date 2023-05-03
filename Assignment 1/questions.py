from utils import *
from bigram_model import *
from unigram_model import *

def q1():
    """
    How many word types (unique words) are there in the training corpus? Please include the end-of-sentence padding
    symbol </s> and the unknown token <unk>. Do not include the start of sentence padding symbol <s>.
    :return:
    """
    with open('./Data/pre_processed_training_data.txt') as training_corpus:
        training_data = file_to_list_list(training_corpus)
        token_counter = count_token(training_data)
        del token_counter['<s>']
        unique_token_total = len(token_counter.keys())
        print('Q1. Total word types in training corpus is:', unique_token_total)
        return unique_token_total


def q2():
    """
    How many word tokens are there in the training corpus? Do not include the start of sentence padding symbol <s>.
    :return:
    """
    with open('./Data/pre_processed_training_data.txt') as training_corpus:
        training_data = file_to_list_list(training_corpus)
        token_counter = count_token(training_data)
        del token_counter['<s>']
        token_total = sum(token_counter.values())
        print('Q1. Total word tokens in training corpus is:', token_total)
        return token_total


def q3():
    """
    What percentage of word tokens and word types in the test corpus did not occur in training (before you mapped the
    unknown words to <unk> in training and test data)? Please include the padding symbol </s> in your calculations.
    Do not include the start of sentence padding symbol <s>.
    :return:
    """
    with open('./Data/padded_and_lowered_training_corpus.txt') as training_corpus:
        with open('./Data/padded_and_lowered_test_corpus.txt') as test_corpus:
            training_data = file_to_list_list(training_corpus)
            test_data = file_to_list_list(test_corpus)
            training_token_counter = count_token(training_data)
            test_token_counter = count_token(test_data)

            del training_token_counter['<s>']
            del test_token_counter['<s>']

            test_token_type_total = len(test_token_counter.keys())
            token_type_list_in_test_not_training = [token
                                                    for token in test_token_counter.keys()
                                                    if token not in training_token_counter]

            percentage_token_type_in_test_not_training = len(
                token_type_list_in_test_not_training) / test_token_type_total

            test_token_total = sum(test_token_counter.values())
            token_count_in_test_not_training = sum([test_token_counter[token]
                                                    for token in token_type_list_in_test_not_training])
            percentage_token_in_test_not_training = token_count_in_test_not_training / test_token_total

            print("The percentage of word tokens in test corpus but not in training corpus is:",
                  percentage_token_in_test_not_training * 100)
            print("The percentage of word types in test corpus but not in training corpus is:",
                  percentage_token_type_in_test_not_training * 100)

            return percentage_token_in_test_not_training, percentage_token_type_in_test_not_training


def q4():
    """
    Now replace singletons in the training data with <unk> symbol and map words(in the test corpus) not observed in
    training to <unk>. What percentage of bigrams (bigram types and bigram tokens) in the test corpus did not occur
    in training ( treat <unk> as a regular token that has been observed). Please include the padding symbol </s> in
    your calculations. Do not include the start of sentence padding symbol <s>.
    :return:
    """

    # get total bigram token type count of test = unigram_token_types ** 2
    # get total bigram token count of test = sum of ngram_token_counter.values()
    # count how many have occurred, since our model only record the token occurred
    # (test_total - test_occurred) / test_total = ans

    bigram_model = load_language_model('./Models/bigram_model.json')
    with open('./Data/test_token_not_in_training_to_unk.txt') as pre_processed_test:
        test_data_list = file_to_list_list(pre_processed_test)
        unigram_test_counter = count_token(test_data_list)
        bigram_test_counter = ngram_count_token(test_data_list, ngram=2, skip=1)
        del unigram_test_counter['<s>']

        # 
        total_bigram_test_type = len(unigram_test_counter.keys()) ** 2
        total_bigram_test_token = sum(bigram_test_counter.values())

        occurred_types = set()
        occured_tokens = 0

        for bigram_test_token_tuple in bigram_test_counter.keys():
            bigram_test_token = f'{bigram_test_token_tuple[0]} {bigram_test_token_tuple[1]}'
            
            if bigram_test_token in bigram_model:
                occured_tokens += bigram_test_counter[bigram_test_token_tuple]
                occurred_types.add(bigram_test_token)

        not_occurred_type_percentage = (total_bigram_test_type - len(occurred_types)) / total_bigram_test_type
        not_occurred_token_percentage = (total_bigram_test_token - occured_tokens) / total_bigram_test_token

        print("The percentage of word tokens in test corpus but not in training corpus is:",
              not_occurred_token_percentage * 100)
        print("The percentage of word types in test corpus but not in training corpus is:",
              not_occurred_type_percentage * 100)
q4()

def q5():
    data = [['i', 'look', 'forward', 'to', 'hearing', 'your', 'reply', '.', '</s>']]
    unigram_prob = unigram_evaluate_log_prob(data, './Models/unigram_model1.json')

    print()
    bigram_prob = bigram_evaluate_log_prob(data, './Models/bigram_model1.json')
    print()
    bigram_prob_add_one_smoothing = bigram_evaluate_log_prob_add_one_smoothing(data,
                                                                               './Data/pre_processed_training_data.txt',
                                                                               skip=1)

    print()
    print('Unigram Log Probability:', unigram_prob)
    print('Bigram Log Probability:', bigram_prob)
    print('Bigram with add one smoothing Log Probability:', bigram_prob_add_one_smoothing)


def q7():
    with open('./Data/test_token_not_in_training_to_unk.txt') as test_file:
        test_data = file_to_list_list(test_file)

        unigram_prob = unigram_evaluate_log_prob(test_data, './Models/unigram_model.json')
        bigram_prob = bigram_evaluate_log_prob(test_data, './Models/bigram_model.json')
        bigram_prob_add_one_smoothing = bigram_evaluate_log_prob_add_one_smoothing(test_data, './Data/pre_processed_training_data.txt')

        print('Unigram Log Probability:', unigram_prob)
        print('Bigram Log Probability:', bigram_prob)
        print('Bigram with add one smoothing Log Probability:', bigram_prob_add_one_smoothing)

        total_token_test_data = sum([len(sentence) - 1 for sentence in test_data])
        unigram_ppl = calculate_ppl(unigram_prob, total_token_test_data)
        bigram_ppl = calculate_ppl(bigram_prob, total_token_test_data)
        bigram_add_one_ppl = calculate_ppl(bigram_prob_add_one_smoothing, total_token_test_data)

        print('Unigram Perplexity:', unigram_ppl)
        print('Bigram Perplexity:', bigram_ppl)
        print('Bigram add one Perplexity:', bigram_add_one_ppl)































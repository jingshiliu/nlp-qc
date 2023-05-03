## Instructions

- For questions in 1.3, there is python file named 'questions.py' which include the code for each question. For example, the method q1() is the code for question 1 in section 1.3

- utils.py contains some utility methods to train language models, such as `count_token()`, `ngram_count_token()`, `file_to_list_list()`, etc.. Majority of the methods have docstring written.

- While reading my code, you will see `defaultdict` rather than regular `dict` because the method `get()` in `defaultdict` is very useful and convenient to prevent parameters not shown in the training or test corpus to occupy memories

- In other word, my code handles the parameters with zero probability dynamically rather than save it into counter or model before using it

## To Run the Program

### Preprocessing

You need to preprocess training file and test file before training and evaluating, and the file `pre_process.py` can help you do that

- <b>Preprocess Training data: </b> pre_process_training_data(training_data_path: str, pre_processed_data_path: str)
- <b>Preprocess Test data: </b> pre_process_test_data(data_path: str, save_path: str, pre_processed_training_path)
- <b>Only lower and pad the file (used for q3): </b> lower_and_pad_file(data_path: str, save_path: str)

### Train Langauge Model

#### Unigram Model

Use method in `unigram_model.py`: train_unigram(preprocessed_data_path, model_save_path, skip=0), skip is used to skip the words in each sentence, mainly used to skip `<s>`

#### Bigram Model

Use method in `bigram_model.py`: train_bigram(preprocessed_data_path, model_save_path, skip=0)

#### Bigram Model with add one smoothing

Use method in `bigram_model.py`: bigram_evaluate_log_prob_add_one_smoothing(data_list_list: list[list], training_data_path: str, skip=0), this one does not return model, but create it during evaluation.


### Section 1.3

As stated above, `questions.py` is for section 1.3 questions. Each method in the file contains specific steps




import decimal
import json
import string


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


def preprocess_comment(comment: str) -> str:
    return separate_punctuation(lowercase_sentence(comment))


def compute_prob(comment: str | list, class_recognizer: dict, prior_prob: float) -> decimal.Decimal:
    if type(comment) is str:
        comment = comment.split()

    # the min of a float is around 1e-310, it's very likely to have the prob of the sentence less than this
    prob = decimal.Decimal(prior_prob)
    for word in comment:
        if word not in class_recognizer:
            continue
        prob = prob * decimal.Decimal(class_recognizer[word])
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
        class_2_prob = compute_prob(word_list, self.model[self.class_2], self.model[f"{self.class_2}_prior"])

        return self.class_1 if class_1_prob > class_2_prob else self.class_2


class CommentClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__(path_to_model='./movie_review_BOW.NB')


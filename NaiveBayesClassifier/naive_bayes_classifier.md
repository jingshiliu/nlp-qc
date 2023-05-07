#### Naive Bayes Classifier

1. Question: What class will Naıve Bayes assign to the sentence “I always like foreign films”? Show your work.

![](images/1.png)

```
p( + | I, always, like, foreign, films) = p(+) * p(I, always, like, foreign, films | + )
                                    = p(+) * p( I | + ) * p( always | + ) * p( like | + ) * p( foreign | + ) * p( films | + )
                                    = 0.4 * 0.09 * 0.07 * 0.29 * 0.04 * 0.08
                                    = 2.33856 * 10^(-6)
                                    

p( - | I, always, like, foreign, films) = p(-) * p(I, always, like, foreign, films | - )
                                    = p(-) * p( I | - ) * p( always | - ) * p( like | - ) * p( foreign | - ) * p( films | - )
                                    = 0.6 * 0.16 * 0.06 * 0.06 * 0.15 * 0.11
                                    = 5.7024 * 10^(-6)

By comparing the result of above two expressions, we find the probability of negative is higher. Therefore, the class negative will be assigned to the sentence.
```

2b. Use the following small corpus of movie reviews to train your classifier. Save the parameters of your model in a file called movie-review-small.NB
```
i. fun, couple, love, love comedy 
ii. fast, furious, shoot action
iii. couple, fly, fast, fun, fun comedy 
iv. furious, shoot, shoot, fun action
v. fly, fast, shoot, love action
```

```
Vocabs: {fun, couple, love, fast, furious, shoot, fly}
|Vocabs| = 7    comedy_total_word = 9       action_total_word = 11
p(comedy) = 9/ (9 + 11) = 0.45
p(action) = 11 / (9 + 11) = 0.55

p(fun | comedy) = (3 + 1) / (9 + 7) = 4/16              p(fun | action) = (1 + 1) / (11 + 7) = 2/18
p(couple | comedy) = (2 + 1) / (9 + 7) = 3/16           p(couple | comedy) = (0 + 1) / (11 + 7) = 1/18
p(love | comedy) = (2 + 1) / (9 + 7) = 3/16             p(love | action) = (1 + 1) / (11 + 7) = 2/18
p(fast | comedy) = (1 + 1) / (9 + 7) = 2/16             p(fast | action) = (2 + 1) / (11 + 7) = 3/18
p(furious | comedy) = (0 + 1) / (9 + 7) = 1/16          p(furious | action) = (2 + 1) / (11 + 7) = 3/18
p(shoot | comedy) = (0 + 1) / (9 + 7) = 1/16            p(shoot | action) = (4 + 1) / (11 + 7) = 5/18
p(fly | comedy) = (1 + 1) / (9 + 7) = 2/16               p(fly | action) = (1 + 1) / (11 + 7) = 2/18
```

#### The code that solves this problem. 

preprocess_folder() uses file at "vocab_path" as the set of vocabulary, reads all file in the folder "folder_path", preprocess all file in the folder, and save the feature vector to folder "output_folder". For example, a file at "./data/movie_review_small/action/1.txt" will be preprocessed and the feature vector will be saved to "./preprocessed/movie_review_small/action/1.json"

naive_bayes() uses file at "vocab_path" as the set of vocabulary, train the model using feature vectors from "class_1_folder_path" and "class_2_folder_path", and save the parameters in json format to file at "result_model_path". The structure of json is provided below.

```
def problem_2b():
    preprocess_folder(folder_path="./data/movie_review_small/action",
                      output_folder="./preprocessed/movie_review_small/action",
                      vocab_path="./data/movie_review_small/movie_review_small.vocab"
                      )
    preprocess_folder(folder_path="./data/movie_review_small/comedy",
                      output_folder="./preprocessed/movie_review_small/comedy",
                      vocab_path="./data/movie_review_small/movie_review_small.vocab"
                      )

    naive_bayes(class_1_folder_path="./preprocessed/movie_review_small/action",
                class_2_folder_path="./preprocessed/movie_review_small/comedy",
                result_model_path="./models/movie_review_small.NB",
                class_1="action",
                class_2="comedy",
                vocab_path="./data/movie_review_small/movie_review_small.vocab"
                )
```

The parameters of the model is saved as a json, and below is the content of the json.
```
{"action": 
    {"couple": 0.05555555555555555, 
     "shoot": 0.2777777777777778, 
     "love": 0.1111111111111111, 
     "fly": 0.1111111111111111, 
     "fast": 0.16666666666666666, 
     "fun": 0.1111111111111111, 
     "furious": 0.16666666666666666
     }, 
 "comedy": 
     {"couple": 0.1875, 
      "shoot": 0.0625, 
      "love": 0.1875, 
      "fly": 0.125, 
      "fast": 0.125, 
      "fun": 0.25, 
      "furious": 0.0625
      }, 
 "action_prior": 0.55, 
 "comedy_prior": 0.45, 
 "class_1": "action",
 "class_2": "comedy"
}
```

2c. Test you classifier on the new document below: {fast, couple, shoot, fly}. Compute the most likely class. Report the probabilities for each class.

Hand-writing calculation
``` 
  p(comedy | fast, couple, shoot, fly) 
= p(comedy) * p(fast | comedy) * p(couple | comedy) * p(shoot | comedy) * p(fly | comedy)
= 9/20 * 2/16 * 3/16 * 1/16 * 2/16
= 0.000082397

  p(action | fast, couple, shoot, fly) 
= p(action) * p(fast | action) * p(couple | action) * p(shoot | action) * p(fly | action)
= 11/20 * 3/18 * 1/18 * 5/18 * 2/18
= 0.000157179

p(action | fast, couple, shoot, fly) > p(comedy | fast, couple, shoot, fly) 
0.000157179 > 0.000082397

The document {fast, couple, shoot, fly} will be assign the class: action
```

Code
```
class NaiveBayesClassifier:
    ...
    def classify(self, comment: str):
        comment = preprocess_comment(comment)
        word_list = comment.split()
        class_1_prob = compute_prob(word_list, self.model[self.class_1], self.model[f"{self.class_1}_prior"])
        class_2_prob = compute_prob(word_list, self.model[self.class_2], self.model[f"{self.class_2}_prior"])

        print(self.class_1, "probability is", class_1_prob)
        print(self.class_2, "probability is", class_2_prob)

        return self.class_1 if class_1_prob > class_2_prob else self.class_2
        
def problem_2c():
    comment = "fast, couple, shoot, fly"
    naive_bayes_classifier = NaiveBayesClassifier(path_to_model='./models/movie_review_small.NB')
    class_estimation = naive_bayes_classifier.classify(comment)

    print(f"Class of sentence {comment} is: {class_estimation}")
```

Result printed by the trained classifier
```
action probability is 0.0001571787837219936
comedy probability is 8.23974609375e-05
Class of sentence fast, couple, shoot, fly is: action
```
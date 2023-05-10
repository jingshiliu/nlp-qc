Example of how to use comment_classifier

``` 
from comment_classifier import CommentClassifier

comment_classifier = CommentClassifier()

result = comment_classifier.classify("hello world")

print(result)
```

`comment_classifier.classify("hello world") ` returns either `pos` or `neg`
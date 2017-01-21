from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np
file = open('news_train.txt',  encoding="utf8")
lines = file.readlines()
file.close()
categories = ['science', 'style', 'culture', 'life', 'economics', 'business', 'travel', 'forces', 'media', 'sport']

train_target = []
train_target_names = []
train_text = []

for line in lines:
    list = line.split('\t')
    train_target.append(categories.index(list[0]))
    train_target_names.append(list[0])
    train_text.append(list[2])


text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-5, n_iter=5, random_state=42)),
])
_ = text_clf.fit(train_text,train_target)

# Load data for test
file = open('news_test.txt',  encoding="utf8")
lines = file.readlines()
file.close()
test_text=[]

for line in lines:
    list = line.split('\n')
    test_text.append(list[0])
docs_test = test_text
predicted = text_clf.predict(docs_test)

file = open('news_output.txt', 'w', encoding="utf8")
for i in predicted:
    file.write(categories[i] + '\n')
file.close()

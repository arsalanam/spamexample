###section 1 using count vectorizer (bag if words model) and random forrest

import pandas as pd
d = pd.read_csv("YouTube-Spam-Collection-v1/Youtube01-Psy.csv")

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

dvec = vectorizer.fit_transform(d['CONTENT'])

dshuf = d.sample(frac=1)
d_train = dshuf[:300]
d_test = dshuf[300:]
d_train_att = vectorizer.fit_transform(d_train['CONTENT']) # fit bag-of-words on training set
d_test_att = vectorizer.transform(d_test['CONTENT']) # reuse on testing set
d_train_label = d_train['CLASS']
d_test_label = d_test['CLASS']

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=80)

#this steps feed classifier with trainig set and associated labels of stuff which is spam and which is not
clf.fit(d_train_att, d_train_label)

##print classifier score

print(clf.score(d_test_att, d_test_label))

###print confusion matrix to see false positive and false negitives

from sklearn.metrics import confusion_matrix
pred_labels = clf.predict(d_test_att)
print(confusion_matrix(d_test_label, pred_labels))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, d_train_att, d_train_label, cv=5)
# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


###section 2..This sections shows a pipe line of count vectorizer and random forrest and bigger data set

# load all datasets and combine them
d = pd.concat([pd.read_csv("YouTube-Spam-Collection-v1/Youtube01-Psy.csv"),
               pd.read_csv("YouTube-Spam-Collection-v1/Youtube02-KatyPerry.csv"),
               pd.read_csv("YouTube-Spam-Collection-v1/Youtube03-LMFAO.csv"),
               pd.read_csv("YouTube-Spam-Collection-v1/Youtube04-Eminem.csv"),
               pd.read_csv("YouTube-Spam-Collection-v1/Youtube05-Shakira.csv")])

dshuf = d.sample(frac=1)
d_content = dshuf['CONTENT']
d_label = dshuf['CLASS']

# set up a pipeline
from sklearn.pipeline import Pipeline, make_pipeline
pipeline = Pipeline([
    ('bag-of-words', CountVectorizer()),
    ('random forest', RandomForestClassifier()),
])
pipeline

pipeline.fit(d_content[:1500],d_label[:1500])

print(pipeline.score(d_content[1500:], d_label[1500:]))

print(pipeline.predict(["what a neat video!"]))

print(pipeline.predict(["plz subscribe to my channel"]))

scores = cross_val_score(pipeline, d_content, d_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))






# section 3..This sections shows a pipe line of count vectorizer , TDIF transformer and random forrest
# it also uses grid search to find best paramters for all classifiers and vectorizers using Grid search
# note Gridsearch will enable and disable tdiftransformer via varibale tfidftransformer__use_idf  to
#see if its usage improve cross score validation


# add tfidf
from sklearn.feature_extraction.text import TfidfTransformer
pipeline2 = make_pipeline(CountVectorizer(),
                          TfidfTransformer(norm=None),
                          RandomForestClassifier())

scores = cross_val_score(pipeline2, d_content, d_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# parameter search
parameters = {

    'countvectorizer__max_features': (None, 1000, 2000),
    'countvectorizer__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'countvectorizer__stop_words': ('english', None),
    'tfidftransformer__use_idf': (True, False), # effectively turn on/off tfidf
    'randomforestclassifier__n_estimators': (20, 50, 100)
}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(pipeline2, parameters, n_jobs=-1, verbose=1)

grid_search.fit(d_content, d_label)

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

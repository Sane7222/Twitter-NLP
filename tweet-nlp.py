# https://www.kaggle.com/competitions/nlp-getting-started

import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import LinearSVC

class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, xs, ys = None): return self
    def transform(self, xs):
        def de_tag(t): return re.sub('&amp;', ' ', t)
        def drop_quote_and_hyphen(t): return re.sub(r"\'|\-", '', t)
        def spacify_non_letter_or_digit(t): return re.sub('\W', ' ', t)
        def combine_spaces(t): return re.sub('\s+', ' ', t)

        xs = xs.str.lower()
        xs = xs.apply(de_tag)
        xs = xs.apply(drop_quote_and_hyphen)
        xs = xs.apply(spacify_non_letter_or_digit)
        xs = xs.apply(combine_spaces)

        return xs

# id, keyword (null), location (null), text, target
df = pd.read_csv('train.csv')
x = df['text']
y = df['target']

steps = [
    ('tokenize', Preprocessor()),
    ('vectorize', CountVectorizer(binary= True, ngram_range= (1, 3), max_features= 100000)),
    ('normalize', Normalizer(norm= 'l2')),
    ('classify', RidgeClassifier()),
]

grid = {
    'vectorize__binary': [True, False],
    'vectorize__ngram_range': [(1, 1), (1, 3), (2, 3)],
    'vectorize__max_features': [10000, 100000],
    'normalize__norm':['l1', 'l2', 'max'],
    'classify': [
        #LogisticRegression(),
        RidgeClassifier(),
        #LinearSVC(),
    ],
}

pipe = Pipeline(steps= steps)
pipe.fit(x, y)
'''search = GridSearchCV(estimator= pipe, param_grid= grid, scoring= 'f1', n_jobs= -1).fit(x, y)

print("F1: " + str(search.best_score_))
print("Best params: " + str(search.best_params_))'''

# Submission
'''df_test = pd.read_csv('test.csv')
x_test = df_test['text']

preds = pipe.predict(x_test)

submission_df = pd.DataFrame({
    'id': df_test['id'],
    'target': preds
})

submission_df.to_csv('submission.csv', index= False)'''

# Demonstration
df_test = pd.read_csv('demon.csv')
x_test = df_test['text']

preds = pipe.predict(x_test)

print(preds)

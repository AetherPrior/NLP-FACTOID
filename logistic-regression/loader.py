from joblib import load
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler

path_to_data = "~/Documents/factoid_qa/"

d = {ord(c): None for c in '''[!"#$%&'()*+,./:;<=>?@[\]^_`{|}~]'''} ##to remove punc
d[ord('-')] = ' '

def my_preprocess(s):
  s = s.lower()
  s = s.translate(d)
  return s

q = "who are you?"
a = "I am yee."
q = my_preprocess(q)
a = my_preprocess(a)
Vec = TfidfVectorizer()
vec.transform(q)
vec.transform(a)
print('Done preprocessing')

models = [load('logistic_regression_concat.joblib'),load('logistic_regression_add.joblib'),load('logistic_regression_average.joblib')]

X = "Who are you"
models[1].predict()
train_val = load('train_values.joblib')
test_val = load('test_values.joblib')

print("Train score: %f" %models[0].score(train_val[0],train_val[3]))
print("Test score: %f" %models[0].score(test_val[0],test_val[3])) 


print("Train score: %f" %models[1].score(train_val[1],train_val[4]))
print("Test score: %f" %models[1].score(test_val[1],test_val[4]))



print("Train score: %f" %models[2].score(train_val[2],train_val[5]))
print("Test score: %f" %models[2].score(test_val[2],test_val[5]))


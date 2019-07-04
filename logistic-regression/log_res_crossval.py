'''
save_concat = "logistic_regression_concat.joblib"
save_add = "logistic_regression_add.joblib"
save_average = "logistic_regression_average.joblib"


train_values = "train_values.joblib"

test_values = "test_values.joblib"
'''


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split,KFold

from imblearn.over_sampling import RandomOverSampler

from joblib import dump
import pandas as pd
import numpy as np
import os

#--------MODIFY THE PATH HERE...
path_to_data = "~/Documents/factoid_qa/"

d = {ord(c): None for c in '''[!"#$%&'()*+,./:;<=>?@[\]^_`{|}~]'''} ##to remove punc
d[ord('-')] = ' '

def my_preprocess(s):
  s = s.lower()
  s = s.translate(d)
  return s


'''
def concatenate_csc_matrices_by_columns(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

    return csc_matrix((new_data, new_indices, new_ind_ptr))
'''

#--------AND HERE
train = pd.read_csv(os.path.join(path_to_data,'data/small_data.tsv'), sep='\t', header=None)
print('Done Reading')
print(train.head())
train.iloc[:,2] = train.iloc[:,2].apply(my_preprocess)
train.iloc[:,1] = train.iloc[:,1].apply(my_preprocess)
print('Done preprocessing')

#apply tf-idf to the vectors
#vocabulary is for both Questions and Answers

words = list(train.iloc[:,1].str.split(' ', expand=True).stack().unique())
words.extend(list(train.iloc[:,2].str.split(' ', expand=True).stack().unique()))
words = list(set(words))





y = train.iloc[:,3]

X = train.iloc[:,1:3]

X_tr = X
y_tr = y
#X_Q_tr = Question for training
#X_A_tr = Answer for training
#X_Q_te = Question for testing
#X_A_te = Answer for testing

kf = KFold(n_splits = 5)

X_tr = np.array(X_tr)
y_tr = np.array(y_tr)
#print(X_tr.shape)
kf.get_n_splits(X_tr)

concat_score = []
add_score = []
average_score = []

for train_index, test_index in kf.split(X_tr):
	vect = TfidfVectorizer(vocabulary = words)
	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X_tr[train_index], X_tr[test_index]
	y_train, y_test = y_tr[train_index], y_tr[test_index]

	X_Q_tr = vect.fit_transform(X_train[:,0])
	X_A_tr = vect.fit_transform(X_train[:,1])
	X_Q_te = vect.transform(X_test[:,0])
	X_A_te = vect.transform(X_test[:,1])

	#print(X_Q_tr)
	#print(X_A_tr)
	#print(y_train,y_test)

	X_concat_train = hstack((X_Q_tr, X_A_tr))

	X_add_train = X_Q_tr+X_A_tr

	X_average_train = X_add_train/2

	X_concat_test = hstack((X_Q_te, X_A_te))

	X_add_test = X_Q_te+X_A_te

	X_average_test = X_add_test/2

	#print(X_concat_test)

	#print(y_tr)

	y_add_train = y_train.copy()
	y_avg_train = y_train.copy()
	#to get equal zeroes and ones in order for the machine to actually learn well
	ros = RandomOverSampler(random_state=42)
	#print(X_concat_train.shape, X_add_train.shape)
	X_concat_train,y_train = ros.fit_resample(X_concat_train,y_train)

	X_add_train,y_add_train = ros.fit_resample(X_add_train,y_add_train)

	X_average_train,y_avg_train  = ros.fit_resample(X_average_train,y_avg_train)
	print("Done some other stuff")
	'''
	#using lbgfs
	train_vals = X_concat_train,X_add_train,X_average_train,y_tr,y_add_tr,y_avg_tr
	test_vals = X_concat_test,X_add_test,X_average_test,y_te,y_te,y_te


	dump(train_vals,train_values)
	dump(test_vals,test_values)
	'''

	clf = LogisticRegression(random_state =42,solver='lbfgs',multi_class ='multinomial').fit(X_concat_train,y_train)

	clf_add = LogisticRegression(random_state =42,solver='lbfgs',multi_class ='multinomial').fit(X_add_train,y_add_train)

	clf_avg = LogisticRegression(random_state = 42,solver = 'lbfgs', multi_class = 'multinomial').fit(X_average_train,y_avg_train)

	#y_pred = clf.predict(X_concat_test)
	#print(y_pred)
	concat_score.append(clf.score(X_concat_test,y_test))
	add_score.append(clf_add.score(X_add_test,y_test))
	average_score.append(clf_avg.score(X_average_test,y_test))

print(concat_score)
print(add_score)
print(average_score)

'''
[0.745125, 0.7225625, 0.7439375, 0.7413125, 0.7508125]
[0.782, 0.7735, 0.786375, 0.7796875, 0.786375]
[0.7790625, 0.769375, 0.7830625, 0.773625, 0.7828125]
'''
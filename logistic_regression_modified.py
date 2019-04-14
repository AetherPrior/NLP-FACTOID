from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack,csr_matrix
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler

import pandas as pd
import numpy as np
import os


path_to_data = "~/Documents/abhinav/"
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
train = pd.read_csv(os.path.join(path_to_data,'data/small_data.tsv'), sep='\t', header=None)
print('Done Reading')
print(train.head())
train.iloc[:,2] = train.iloc[:,2].apply(my_preprocess)
train.iloc[:,1] = train.iloc[:,1].apply(my_preprocess)
print('Done preprocessing')
#apply tf-idf to the vectors

#vocabulary is for both Q and A
words = list(train.iloc[:,1].split(' ', expand=True).stack().unique())
words.append(list(train.iloc[:,1].split(' ', expand=True).stack().unique()))
words = list(set(words))
vect = TfidfVectorizer(vocabulary = words)

X_Q = vect.fit_transform(train.iloc[:,1])
X_A = vect.fit_transform(train.iloc[:,2])


print(type(X_Q))
#print(vect.get_feature_names())

print(X_Q[1])
y = train.iloc[:,3]

##concatenate the question and answer vector after tf-idf over all of em

#print(X_A.shape)
#print(X_Q.shape)

length_vect = len(train.iloc[:,2])
X_concat = hstack((X_Q, X_A))

X_concat_train,X_concat_test,y_concat_train,y_concat_test =train_test_split(X_concat, y, test_size=0.2)
#to get equal zeroes and ones in order for the machine to actually learn well
ros = RandomOverSampler(random_state=42)

X_concat_train,y_concat_train = ros.fit_resample(X_concat_train,y_concat_train)

#X_concat_test,y_concat_test = ros.fit_resample(X_concat_test,y_concat_test)

#diff_in_cols = X_A.shape[1]-X_A.shape[0];
#X_stacked_for_sum = hstack((X_Q,csr_matrix(X_Q.shape[0],diff_in_cols)))
#X_sum = X_stacked_for_sum + X_A
#X_avg = X_sum/2
print("Done some other stuff")

print("2D Sparse done")

##using lbgfs
clf    = LogisticRegression(random_state =0,solver='lbfgs',multi_class ='multinomial',max_iter=300).fit(X_concat_train,y_concat_train)
#clfsum = LogisticRegression(random_state =0,solver='lbfgs',multi_class ='multinomial').fit(X_sum,y)
#clfavg = LogisticRegression(random_state =0,solver='lbfgs',multi_class ='multinomial').fit(X_avg,y)
#predict

y_pred = clf.predict(X_concat_test)

#predict_probabiliy
#print(clf.predict_proba(X_concat))
print("Train score: %d" %clf.score(X_concat_train,y_concat_train))
print("Test score: %d" %clf.score(X_concat_test,y_concat_test)) 

#X_sum = X_Q + X_A
#X_avg = (X_Q + X_A)/2

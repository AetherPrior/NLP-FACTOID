from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack,csr_matrix
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler

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


vect = TfidfVectorizer(vocabulary = words)

y = train.iloc[:,3]

#X_Q_tr = Question for training
#X_A_tr = Answer for training
#X_Q_te = Question for testing
#X_A_te = Answer for testing

X_Q_tr,X_Q_te,X_A_tr,X_A_te,y_tr,y_te     = train_test_split(train.iloc[:,1],train.iloc[:,2], y, test_size=0.2)

X_Q_tr = vect.fit_transform(X_Q_tr)
X_A_tr = vect.fit_transform(X_A_tr)

#transform the test vectors

X_Q_te = vect.transform(X_Q_te)
X_A_te = vect.transform(X_A_te)



##concatenate the question and answer vector after tf-idf over all of em


X_concat_train = hstack((X_Q_tr, X_A_tr))

#print(X_concat_train)

X_concat_test = hstack((X_Q_te, X_A_te))

#print(X_concat_test)

#print(y_tr)

#to get equal zeroes and ones in order for the machine to actually learn well
ros = RandomOverSampler(random_state=42)

X_concat_train,y_tr = ros.fit_resample(X_concat_train,y_tr)


print("Done some other stuff")
##using lbgfs
clf = LogisticRegression(random_state =42,solver='lbfgs',multi_class ='multinomial').fit(X_concat_train,y_tr)


#predict

#y_pred = clf.predict(X_concat_test)

#predict_score
print("Train score: %d" %clf.score(X_concat_train,y_tr))
print("Test score: %d" %clf.score(X_concat_test,y_te)) 

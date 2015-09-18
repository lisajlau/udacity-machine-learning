#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



print("=========")
t0 = time()
#########################################################
### your code goes here ###
from sklearn import svm
clf = svm.SVC(kernel="rbf", C=10000.0, gamma=0.0)


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
features_train = features_train[:len(features_train)] 
labels_train = labels_train[:len(labels_train)] 

clf.fit(features_train,labels_train)


#### store your predictions in a list named pred
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print "==== Accuracy ====="
print acc

print "==== Training time ====="
print round(time()-t0, 3), "s"

from numpy  import *
print "====  Total ====="
print pred.sum()
#########################################################



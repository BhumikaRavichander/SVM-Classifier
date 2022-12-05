# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:24:25 2021

@author: bhumi   ----------PROJECT 3----------
"""

import idx2numpy
import numpy as np
import cv2
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import confusion_matrix

file1 =  "train-images.idx3-ubyte"
file2 =  "train-labels.idx1-ubyte"

train_images = idx2numpy.convert_from_file(file1)
train_labels = idx2numpy.convert_from_file(file2)

data = train_images.astype('float')
data = data/255




###############   Task 1   ############

all_accuracy = []


X = data.reshape(data.shape[0],data.shape[1]*data.shape[2])
print(np.shape(X))


clf = svm.SVC(kernel ='rbf')

# Train on the first 10000 images:
train_x = X[:1000]
train_y = train_labels[:1000]

print("Train model")
clf.fit(train_x, train_y)

# Test on the next 1000 images:
test_x = X[2000:2100]
expected1 = train_labels[2000:2100].tolist()
print("Compute predictions")
predicted1 = clf.predict(test_x)
print("====================  task-1 ==========================")
print("Accuracy: ", accuracy_score(expected1, predicted1))
acc1 =accuracy_score(expected1, predicted1)
all_accuracy.append(acc1)





################  Task 2  ###################


clf = svm.SVC(kernel ='rbf')

# Train on the first 10000 images:
train_x = X[:10000]
train_y = train_labels[:10000]

print("Train model")
clf.fit(train_x, train_y)

# Test on the next 100 images:
test_x = X[20000:20100]
expected2 = train_labels[20000:20100].tolist()
print("Compute predictions")
predicted2 = clf.predict(test_x)

print("====================  task-2 ==========================")
print("Accuracy: ", accuracy_score(expected2, predicted2))

acc2 =accuracy_score(expected2, predicted2)
all_accuracy.append(acc2)






##############   Task 3   #################

# Train on the first 10000 images:
train_x = X[:10000]
train_y = train_labels[:10000]

print("Train model")
clf.fit(train_x, train_y)

# Test on the next 1000 images:
test_x = X[20000:21000]
expected3 = train_labels[20000:21000].tolist()
print("Compute predictions")
predicted3 = clf.predict(test_x)

print("====================  task-3 ==========================")
print("Accuracy: ", accuracy_score(expected3, predicted3))
acc3 =accuracy_score(expected3, predicted3)
all_accuracy.append(acc3)

print("====================  Confusion Matrix of task 1,2 ,3 ==========================")
print(confusion_matrix(expected1, predicted1))
print(confusion_matrix(expected2, predicted2))
print(confusion_matrix(expected3, predicted3))



##################   TASK 5  ################


clf = svm.SVC(kernel ='poly')

train_x = X[:10000]
train_y = train_labels[:10000]

print("Train model")
clf.fit(train_x, train_y)

# Test on the next 1000 images:
test_x = X[20000:21000]
expected4 = train_labels[20000:21000].tolist()
print("Compute predictions")
predicted4 = clf.predict(test_x)

print("====================  task-4 ==========================")
print("Accuracy: ", accuracy_score(expected4, predicted4))
acc4 =accuracy_score(expected4, predicted4)
all_accuracy.append(acc4)
print(confusion_matrix(expected4, predicted4))






###################  TASK 6  ##################


train_images = idx2numpy.convert_from_file(file1)
train_labels = idx2numpy.convert_from_file(file2)

X = train_images.reshape(train_images.shape[0],train_images.shape[1]*train_images.shape[2])
input_threshold = []

i=0
for i in range(len(X)):
    normalise  = np.where(X[i] >120,1,0)
    input_threshold.append(normalise)
print(np.shape(input_threshold))



clf = svm.SVC(kernel ='poly')

train_x = input_threshold[:10000]
train_y = train_labels[:10000]

print("Train model")
clf.fit(train_x, train_y)

# Test on the next 1000 images:
test_x = input_threshold[20000:21000]
expected5 = train_labels[20000:21000].tolist()
print("Compute predictions")
predicted5 = clf.predict(test_x)
print("====================  task-5 ==========================")
print("Accuracy: ", accuracy_score(expected5, predicted5))
acc5 =accuracy_score(expected5, predicted5)
all_accuracy.append(acc5)
print(confusion_matrix(expected5, predicted5))



####################   TASK 7  ################

print("====================  Accuracy All ==========================")

print(all_accuracy)





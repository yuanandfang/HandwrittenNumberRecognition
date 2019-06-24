#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: suQing_ time:2019/6/20
import time
from sklearn.externals import joblib
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Softmax_test import softmax

print('Start load data')

time_1 = time.time()
plt.rcParams['figure.figsize'] = (20., 20.)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#加载数据集
data = pd.read_csv('D:/MNIST.csv', header=0).values
imgs = data[0::, 1::]
labels = data[::, 0]
time_2 = time.time()
print('load data cost ' + str(time_2 - time_1) + ' second')


'''
classes = range(10)
num_classes = len(classes)
samples_per_class = 15
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(labels == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(imgs[idx].reshape(28,28).astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()
'''
#划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.33, random_state=40)

#print(X_train.shape)
#print(X_test.shape)

print("Start training")
discern = softmax()
discern.train(X_train, y_train)
'''保存训练模型
joblib.dump(discern,"Train_model.model")
'''
print("Complete the training")

print("Start predict")
y_predict = discern.predict(X_test)
print('Complete the predict')

score = accuracy_score(y_test, y_predict)
print("The accruacy socre is " + str(score))





#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: suQing time:2019/6/21
#已知模型预测
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_predict
import matplotlib.pyplot as plt

#加载model
knownmodel = joblib.load("Train_model.model")
#加载数据集
data = pd.read_csv('D:/MNIST.csv', header=0).values
imgs = data[0::, 1::]
labels = data[::, 0]

#分配测试集
X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.33, random_state=9527)

#测试评分
y_predict = knownmodel.predict(X_test)

score = accuracy_score(y_test, y_predict)
print("The accruacy socre is " + str(score))

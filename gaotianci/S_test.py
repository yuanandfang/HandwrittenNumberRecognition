#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: suQing time:2019/6/21

from sklearn.externals import joblib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("D:/test_13.jpg")
#处理图片
iml = np.array(img, 'f')
nm_arr = iml.reshape([1, 784])
nm_arr = nm_arr.astype(np.float32)
img_ready = np.multiply(nm_arr, 1.0/255.0)
#加载模型预测
Mnist = joblib.load("Train_model.model")
discern = Mnist.predict(img_ready)
print (discern)

#test3.jpg预测失败
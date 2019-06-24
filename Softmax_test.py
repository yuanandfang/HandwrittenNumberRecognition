#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: suQing_ time:2019/6/20

from __future__ import print_function
import math
import pandas as pd
import numpy as np
import random



class softmax(object):
    def __init__(self):
        self.learning_step = 0.000001
        self.max_iteration = 100001
        self.weight_lambda = 0.01

    def func_1(self, x, l):
        theta = self.w[l]
        product = np.dot(theta, x)

        return math.exp(product)

    def func_2(self,x,j):
        molecule = self.func_1(x, j)
        denominator = sum([self.func_1(x, i)for i in range(self.k)])
        quotients = molecule/denominator

        return quotients

    def func_3(self, x, y, j):
        front = int(y == j)
        back = self.func_2(x, j)
        weight = self.weight_lambda*self.w[j]
        cons = weight-x*(front-back)

        return cons

    def train(self, features, labels):
        self.k = len(set(labels))
        self.w = np.zeros((self.k, len(features[0])+1))
        time = 0
        k = 0
        while time < self.max_iteration:
            i = time/1000
            if i >= k and i <= 100:
                print("\rTrained %d %%" % i, end='')
                k += 1
            time += 1
            index = random.randint(0, len(labels) - 1)
            x = features[index]
            y = labels[index]
            x = list(x)
            x.append(1.0)
            x = np.array(x)
            derivatives = [self.func_3(x, y, p) for p in range(self.k)]
            for q in range(self.k):
                self.w[q] -= self.learning_step * derivatives[q]
        print('')

    def max(self,x):
        cons = np.dot(self.w, x)
        row, column = cons.shape
        position = np.argmax(cons)
        max, need = divmod(position, column)

        return max

    def predict(self,features):
        labels = []
        for features in features :
            x = list(features)
            x.append(1)
            x = np.matrix(x)
            x = np.transpose(x)
            labels.append(self.max(x))
        return labels

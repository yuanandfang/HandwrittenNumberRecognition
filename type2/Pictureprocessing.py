#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: suQing_ time:2019/6/20

import numpy as np
import matplotlib.pyplot as plt

# 图片二值化

from PIL import Image

img = Image.open('D:/test0.jpg')

Img = img.convert('L')
Img.save("D:/test1.jpg")
threshold = 150
table = []
for i in range(256):
    if i < threshold:
        table.append(1)
    else:
        table.append(0)
photo = Img.point(table, '1')
photo.save("D:/test2.jpg")
out = photo.resize((28,28))
out.save("D:/test3.jpg")


print(type(threshold))

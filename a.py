# -*- coding: utf-8 -*-
# @Time    : 2021/10/18 22:26
# @Author  : lpd
# @File    : a.py
import glob

for id in glob.glob('data/imgs/*.jpg'):
    print(id)
    name = id.split('\\')[-1].split('.')[0]
    print(name)
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:27:07 2019

@author: dari
"""

import numpy as np
import pandas as pd
import cv2 as cv
import os

# Create dirs
dirs1 = ['Training', 'PublicTest', 'PrivateTest']
dirs2 = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']
for dir1 in dirs1:
    for dir2 in dirs2:
        if not os.path.exists('{}/{}'.format(dir1, dir2)):
            os.makedirs('{}/{}'.format(dir1, dir2))
# Create dirs end

data = pd.read_csv('fer2013.csv')

stryuzler = data['pixels'].values

counts = []

for i in range(len(dirs2)):
    counts.append(1)

dosya = ''

for i, stryuz in enumerate(stryuzler):
    a = stryuz.split(' ')
    b = np.asarray(a, 'float').reshape(48, 48, 1)
    ad = ''
    #    sinirli
    emo = data['emotion'][i]
    dataType = data['Usage'][i]

    if (emo == 0):
        dosya = 'Angry'
        ad = counts[0]
        counts[0] = counts[0] + 1
    elif (emo == 1):
        dosya = 'Disgust'
        ad = counts[1]
        counts[1] = counts[1] + 1
    elif (emo == 2):
        dosya = 'Fear'
        ad = counts[2]
        counts[2] = counts[2] + 1
    elif (emo == 3):
        dosya = 'Happy'
        ad = counts[3]
        counts[3] = counts[3] + 1
    elif (emo == 4):
        dosya = 'Sad'
        ad = counts[4]
        counts[4] = counts[4] + 1
    elif (emo == 5):
        dosya = 'Surprised'
        ad = counts[5]
        counts[5] = counts[5] + 1
    elif (emo == 6):
        dosya = 'Neutral'
        ad = counts[6]
        counts[6] = counts[6] + 1
    if not os.path.exists('{}/{}/{}.jpg'.format(dataType, dosya, ad)):
        cv.imwrite('{}/{}/{}.jpg'.format(dataType, dosya, ad), b)

print('Mission completed :) !')


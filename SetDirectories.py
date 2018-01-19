#!/usr/bin/python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#get_ipython().magic('matplotlib inline')

path_animals = "./ImageNet/Animals/"
path_junk = "./ImageNet/Junk/"

path_felines = "./ImageNet/Felines/"
path_canines = "./ImageNet/Canines/"

path_0 = path_canines
path_1 = path_felines
###=============================####

PathToTask = './Task_2'

os.makedirs('%s/train/0'%PathToTask)
os.mkdir('%s/train/1'%PathToTask)
os.makedirs('%s/val/0'%PathToTask)
os.mkdir('%s/val/1'%PathToTask)

def assignImages(path_in, path_train, path_val):
    imagesList = os.listdir(path_in) #List of images in specified directory
    L = len(imagesList)
    validation_sample = np.random.choice(range(L), round(L*0.1)) #10% of total samples go to validation

    for i,image in enumerate(imagesList):
        img = cv2.imread(path_in+image)
        if i in validation_sample:
            cv2.imwrite(path_val+image, img)
        else:
            cv2.imwrite(path_train+image, img)

assignImages(path_0, '%s/train/0/'%PathToTask, '%s/val/0/'%PathToTask)
assignImages(path_1, '%s/train/1/'%PathToTask, '%s/val/1/'%PathToTask)

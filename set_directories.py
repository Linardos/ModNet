#!/usr/bin/python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#get_ipython().magic('matplotlib inline')

path_animals = "./ImageNet/Animals/"
path_not_animals = "./ImageNet/Junk/"

path_felines = "./ImageNet/Felines/"
path_canines = "./ImageNet/Canines/"

path_not_felines = "./ImageNet/NotFelines/"
path_not_canines = "./ImageNet/NotCanines/"

path_wolf = "./ImageNet/Wolf/"
path_not_wolf = "./ImageNet/Animals_Not_Wolf/"

path_0 = path_not_animals
path_1 = path_animals
PathToTask = './Task_1_Animals/'

###=============================####

def main():

    os.makedirs('%s/train/0'%PathToTask)
    os.mkdir('%s/train/1'%PathToTask)
    os.makedirs('%s/val/0'%PathToTask)
    os.mkdir('%s/val/1'%PathToTask)

    assignImages(path_0, path_train = '%s/train/0/'%PathToTask, path_val = '%s/val/0/'%PathToTask)
    assignImages(path_1, path_train = '%s/train/1/'%PathToTask, path_val = '%s/val/1/'%PathToTask)

def assignImages(path_in, path_train, path_val):

    imagesList = os.listdir(path_in) #List of images in specified directory
    L = len(imagesList)
    validation_sample = np.random.choice(range(L), round(L*0.2), replace=False) #10% of total samples go to validation

    for i,image in enumerate(imagesList):
        img = cv2.imread(path_in+image)
        if i in validation_sample:
            cv2.imwrite(path_val+image, img)
        else:
            cv2.imwrite(path_train+image, img)

def shuffle(path_train, path_val, val_ratio=0.2):

    os.system('mv %s* %s'%(path_val, path_train)) #move everything to train

    images = np.array(os.listdir(path_train)) #List of images in specified directory
    L = len(images)
    validation_samples = np.random.choice(range(L), round(L*val_ratio), replace=False) #10% of total samples go to validation

    for image in images[validation_samples]:
        os.system('mv %s%s %s'%(path_train, image, path_val))


if __name__ == '__main__':
    main()

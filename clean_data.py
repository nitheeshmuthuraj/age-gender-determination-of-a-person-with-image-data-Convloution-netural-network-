# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:57:49 2018

@author: n.muthuraj
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:51:46 2018

@author: n.muthuraj
"""


import glob
import numpy as np
import pandas as pd
import cv2


def get_file_name_age():
    file_names=np.array(pd.read_csv('file_name.csv'))
    age=np.array(pd.read_csv('age_modified_latest.csv'))
    list_images=glob.glob("*.jpg")
   
    list_file_name=[]
    for i in range(len(age)):
        l=[]
        for j in file_names[i]:
            l.append(str(j))
        list_file_name.append(''.join(l).split('na')[0])
    labels=[]
    for index in range(len(list_images)):
        labels.append((age[list_file_name.index(list_images[index])]))
    return list_images,labels


def age_check():
#    print("age which is greater than 90 or less than 5")
    for index in range(len(labels)):
        if(labels[index][0] > 90 or labels[index][0]<5 ):            
            pass
#            print(list_images[index],":",labels[index])

def ensure_all_3_channels():
#    print("ensuring all are color images")
    for index in range(len(list_images)):
        image_data=cv2.imread(list_images[index])
        try:
            if(np.shape(image_data)[2] != 3):
#                print(list_images[index])
                pass
        except:
            pass
#            print("something is wrong with this image")
#            print (list_images[index])
#    print("all chanels ensured")

list_images,labels=get_file_name_age()
age_check()
ensure_all_3_channels()

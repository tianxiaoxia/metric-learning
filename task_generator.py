# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 15:00:25 2018

@author: alex
"""
import os
import random


def omniglot_character_folders():
    data_folder = 'C:\\Users\\alex\\metric-learning\\datas\\omniglot_resized\\'

    character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
    random.seed(1)
    random.shuffle(character_folders)

    num_train = 1200
    metatrain_character_folders = character_folders[:num_train]
    metaval_character_folders = character_folders[num_train:]

    return metatrain_character_folders,metaval_character_folders

def get_class(sample):
    return os.path.join(*sample.split('\\')[:-1])

def omniglotTask(character_folders, num_classes, support_set_num, target_set_num): 
    # character_folders denotes metatrain_character_folders or metaval_character_folders
    class_folders = random.sample(character_folders, num_classes)  
    # print(class_folders)
    labels = np.array(range(len(class_folders)))
    labels = dict(zip(class_folders, labels))
    
    samples = {}
    support_set = []
    target_set = []
    
    for c in class_folders:
        temp = [os.path.join(c, x) for x in os.listdir(c)]  
        samples[c] = random.sample(temp, len(temp))   

        support_set += samples[c][:support_set_num]
        target_set += samples[c][support_set_num:support_set_num + target_set_num]
    
    support_set_labels = [labels[get_class(x)] for x in support_set]
    target_set_labels = [labels[get_class(x)] for x in target_set]
    
    return support_set,target_set,support_set_labels,target_set_labels
    
    

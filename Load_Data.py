import os
from tensorflow import keras
import numpy as np
import random
from PIL import Image

base_dir = os.getcwd()

def load_image_dataset_from_directory(main_dir, train_dir, valid_dir, test_dir):
    train_ds = keras.utils.image_dataset_from_directory(
        directory=base_dir+main_dir+train_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        seed=42,    # validation_split을 위한 seed 고정
        validation_split = 0.2,
        subset="training",
        )
    valid_ds = keras.utils.image_dataset_from_directory(
        directory=base_dir+main_dir+train_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        seed=42,    # validation_split을 위한 seed 고정
        validation_split = 0.2,
        subset="validation",
        )
    test_ds = keras.utils.image_dataset_from_directory(
        directory=base_dir+main_dir+test_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        )
    return train_ds, valid_ds, test_ds



def load_image_batch_size_one(
    main_dir, train_dir,
    label_names,
    target_size=None,
    interpolation='nearest',
    keep_aspect_ratio=False
    ):
    num_of_labels = len(label_names)
    label_list = {}
    for i, label_name in enumerate(label_names):
        label = np.zeros(num_of_labels)
        label[i] = 1
        #label_list[label_name] = label
        label_list[label_name] = i

    batch_set = []
    for label_name in label_names:
        label_dir = f'/{label_name}'
        for i in range(1,51):
            if i < 10:
                file_num = '0'+str(i)
            else:
                file_num = str(i)
            file_name = label_name+file_num
            this_image = keras.preprocessing.image.load_img(
                path=base_dir+main_dir+train_dir+label_dir+'/'+file_name+'.jpg',
                target_size=target_size,
                interpolation=interpolation,
                keep_aspect_ratio=keep_aspect_ratio)
            image_arr = keras.preprocessing.image.img_to_array(this_image)
            batch_set.append([image_arr, label_list[label_name]])
    
    return batch_set
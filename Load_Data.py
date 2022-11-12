import os
from tensorflow import keras

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
        directory=base_dir+main_dir+valid_dir,
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
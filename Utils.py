import os
import shutil
import statistics

import tensorflow as tf
import cv2.cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from keras.utils import np_utils
from numpy import asarray
from sklearn.preprocessing import LabelEncoder


def avg_std(data, name):
    print(f'Results for {name}')
    print(f"Average: {round(statistics.mean(data), 4)}")
    print(f"Standard Dev: {round(statistics.stdev(data), 4)}")
    print()


def clean_data_for_labels(data):
    data.drop(axis=1, labels=['xmin', 'ymin', 'width', 'height'], inplace=True)
    enc = LabelEncoder()
    enc.fit(data['class'])
    return data.drop(axis=1, labels=['class']), np_utils.to_categorical(enc.transform(data['class']), 3)


def clean_image_directory(data, base: str):
    """
    Moves files to a separate directory that pass the filtering phase(s) of the data.
    :param data:
    :param base: Base directory of where the image is
    """

    file_count = 0
    file_index = 0
    saved_files = f'Images_Cleaned'
    if not os.path.exists(saved_files):
        os.mkdir(saved_files)
    else:
        shutil.rmtree(saved_files)
        os.mkdir(saved_files)
    new_data = data.copy()
    for file in data.values:
        try:
            name = f'{file[0]}.jpg'
            if os.path.exists(f'{base}/{name}'):
                min_x = file[2]
                min_y = file[3]
                max_x = file[2] + file[4]
                max_y = file[3] + file[5]
                img = Image.open(f'{base}/{name}')
                new_image = img.crop(box=(min_x, min_y, max_x, max_y))
                if new_image.size[0] < 100 or new_image.size[1] < 100 or remove_noisy_image(f'{base}/{name}'):
                    new_data.drop(labels=file_index, axis=0, inplace=True)
                else:
                    new_image = new_image.resize((200, 200))
                    new_image.save(f'{saved_files}/{file[0]}.jpg')
                    file_count += 1
                    file_count, new_data = rotate_and_save(new_image, file, saved_files, file_count, new_data)

                    new_image = new_image.rotate(90)
                    new_image = new_image.transpose(method=Image.FLIP_LEFT_RIGHT)
                    save_image(new_image, f'{saved_files}/{file[0]}_INVERT_{file_count}.jpg')
                    new_data = append_data_frame(original=new_data, columns=list(data.columns),
                                                 data_frame=[f'{file[0]}_INVERT_{file_count}', file[1], file[2],
                                                             file[3],
                                                             file[4], file[5]])
                    file_count += 1
        except:
            pass
        file_index += 1
    return file_count, new_data


def remove_noisy_image(image_location):
    """
    Checks if the image is blurred.
    :param image_location:
    :return: whether the image is blurred or not
    """
    return 100 > cv.Laplacian(src=cv.imread(image_location), ddepth=cv.CV_32F).var()


def log(log_type: str, message: str):
    print(f'[{log_type.upper()}]\t{message}')


def append_data_frame(original, data_frame, columns):
    return original.append(pd.DataFrame(data=[data_frame], columns=columns), ignore_index=True)


def save_image(image, name):
    image.save(name)


def rotate_and_save(image, file, directory, file_count, new_data):
    rotation = 90
    for i in range(1, 4):
        image = image.rotate(90)
        save_image(image, f'{directory}/{file[0]}_{str(rotation * i)}_{file_count}.jpg')
        new_data = append_data_frame(original=new_data, columns=list(new_data.columns),
                                     data_frame=[f'{file[0]}_{str(rotation * i)}_{file_count}', file[1], file[2], file[3], file[4],
                                                 file[5]])
        file_count += 1
    return file_count, new_data


def convert_to_array(data, data_type):
    if data_type == "Train" or data_type == "Validation":
        data_type = "Images_Cleaned"
    else:
        data_type = "Test_Images_Cleaned"
    index = 0
    new_data = []
    for file in data.values:
        image = Image.open(f'{data_type}/{file}.jpg')
        new_data.append(tf.keras.preprocessing.image.img_to_array(image)/255)

        index += 1

    return np.array(new_data)

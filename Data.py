# imports
import os

import pandas as pd

from Utils import clean_image_directory, log


class Data:
    def __init__(self, base='', file=''):
        data_type = file.split(".")[0]
        self.data = None
        if file != '':
            self.set_base_dir(base)
            self.set_file_name(file)
            self.set_data_path(path=self.get_file_path())
        else:
            self.base_dir = None
            self.data_file_name = None
        self.cnn = None
        self.clean_data(file_type=data_type)
        return

    def get_base_dir(self):
        return self.base_dir

    def set_base_dir(self, base=''):
        self.base_dir = base

    def get_file_name(self):
        return self.data_file_name

    def set_file_name(self, file_name=''):
        self.data_file_name = file_name

    def set_data_path(self, path):
        if not os.path.exists(path):
            log("error", "File does not exist!")
            return
        self.set_data_frame(data=pd.read_csv(path))

    def get_file_path(self):
        return f'{self.get_base_dir()}/{self.get_file_name()}'

    def get_data_frame(self):
        return self.data

    def set_data_frame(self, data):
        self.data = data

    def clean_data(self, file_type):
        # enc = LabelEncoder()
        # enc.fit(self.data['class'])
        # c_data = np_utils.to_categorical(enc.transform(self.data['class']))
        if file_type == "Train":
            image_dir = "Train_Images"
        else:
            image_dir = "Test_Images"
        count = 0
        file_index = 0
        for file in self.data.values:
            if not os.path.exists(f'{image_dir}/{file[0]}.jpg'):
                self.data.drop(index=file_index, inplace=True)
                count += 1
            file_index += 1
        log("info", f'Removed {count} Images From Dataset')
        count, new_data = clean_image_directory(data=self.data, base=image_dir)
        log("info", f'Moved {count} Images To The Cleaned Directory')
        self.set_data_frame(data=new_data)

    def __str__(self):
        return f'{self.get_file_path()}\n{self.get_data_frame()}'

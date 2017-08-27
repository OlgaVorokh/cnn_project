# -*- coding: utf-8 -*-

import os


class BaseConfig(object):
    def __init__(self):
        self.input_filepath = None
        self.output_filepath = None
        self.source_directory = None
        self.data_directory = '../data/'

    def _make_path(self, path, name):
        return os.path.join(path, name)


class EnglishLiesTestConfig(BaseConfig):
    def __init__(self):
        super(EnglishLiesTestConfig, self).__init__()
        self.source_directory = self._make_path(self.data_directory, 'english')
        self.input_filepath = self._make_path(self.source_directory, 'test_parsed.txt')
        self.output_filepath = self._make_path(self.source_directory, 'test_sentenses.txt')


class EnglishLiesTrainConfig(BaseConfig):
    def __init__(self):
        super(EnglishLiesTrainConfig, self).__init__()
        self.source_directory = self._make_path(self.data_directory, 'english')
        self.input_filepath = self._make_path(self.source_directory, 'train_parsed.txt')
        self.output_filepath = self._make_path(self.source_directory, 'train_sentenses.txt')


class EnglishTrainConfig(BaseConfig):
    def __init__(self):
        super(EnglishTrainConfig, self).__init__()
        self.source_directory = self._make_path(self.data_directory, 'english_big')
        self.input_filepath = self._make_path(self.source_directory, 'train_parsed.txt')
        self.output_filepath = self._make_path(self.source_directory, 'train_sentenses.txt')


class EnglishTestConfig(BaseConfig):
    def __init__(self):
        super(EnglishTestConfig, self).__init__()
        self.source_directory = self._make_path(self.data_directory, 'english_big')
        self.input_filepath = self._make_path(self.source_directory, 'test_parsed.txt')
        self.output_filepath = self._make_path(self.source_directory, 'test_sentenses.txt')


class EnglishPartutTrainConfig(BaseConfig):
    def __init__(self):
        super(EnglishPartutTrainConfig, self).__init__()
        self.source_directory = self._make_path(self.data_directory, 'english_partut')
        self.input_filepath = self._make_path(self.source_directory, 'train_parsed.txt')
        self.output_filepath = self._make_path(self.source_directory, 'train_sentenses.txt')


class EnglishPartutTestConfig(BaseConfig):
    def __init__(self):
        super(EnglishPartutTestConfig, self).__init__()
        self.source_directory = self._make_path(self.data_directory, 'english_partut')
        self.input_filepath = self._make_path(self.source_directory, 'test_parsed.txt')
        self.output_filepath = self._make_path(self.source_directory, 'test_sentenses.txt')


class Word2VecConfig(BaseConfig):
    def __init__(self):
        super(Word2VecConfig, self).__init__()
        self.source_directory = self._make_path(self.data_directory, 'word2vec')
        self.input_filepath = self._make_path(self.source_directory, 'text8')
        self.output_filepath = self._make_path(self.source_directory, 'data.txt')
        self.vectors = self._make_path(self.source_directory, 'vectors1.txt')


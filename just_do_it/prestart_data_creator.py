# -*- coding: utf-8 -*-


from . import helper_file, config


class PreStartDataCreator(object):
    def __init__(self):
        self.configuration_word2vec = config.Word2VecConfig()

        self.configurations = [
            config.EnglishLiesTrainConfig(),
            config.EnglishLiesTestConfig(),
            config.EnglishLiesTrainConfig(),
            config.EnglishPartutTestConfig(),
            config.EnglishTrainConfig(),
            config.EnglishTestConfig(),
        ]

    def make_word2vec_input(self):
        self._get_words_only()
        self._merge_files()
        # after that train_word2vec.sh should be made

    def _get_words_only(self):
        file_getter = helper_file.FileWordsMaker()
        for configuration in self.configurations:
            file_getter.get_words(
                data_filename=configuration.input_filepath,
                result_filename=configuration.output_filepath,
            )

    def _merge_files(self):
        merger = helper_file.FileMerger()
        files_lst = [c.output_filepath for c in self.configurations]
        files_lst.append(self.configuration_word2vec.input_filepath)
        merger.merge(
            result_filepath=self.configuration_word2vec.output_filepath,
            files_lst=files_lst,
        )

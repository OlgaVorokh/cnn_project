# -*- coding: utf-8 -*-

from .helper_sentense import SentenceGetter


class FileWordsMaker(object):
    def get_words(self, data_filename, result_filename):
        sentence_getter = SentenceGetter(filename=data_filename)
        sentence_generator = sentence_getter()

        with open(result_filename, 'a') as f:
            for sentence_parser in sentence_generator:
                words = []
                for index in xrange(len(sentence_parser)):
                    if sentence_parser[index].upostag == u'PUNCT':
                        continue
                    words.append(sentence_parser[index].form.lower())

                to_file = u' '.join(words)
                f.write(to_file.encode('utf-8') + '\n')


class FileMerger(object):
    def merge(self, result_filepath, files_lst):
        for input_filepath in files_lst:
            lines = self._get_lines(input_filepath)
            self._write_to_result_file(result_filepath, lines)

    def _get_lines(self, filename):
        with open(filename, 'r') as f:
            return f.readlines()

    def _write_to_result_file(self, result_filepath, lines):
        with open(result_filepath, 'a') as f:
            for line in lines:
                decode_line = line.decode('utf-8').strip()
                f.write(decode_line.encode('utf-8') + '\n')

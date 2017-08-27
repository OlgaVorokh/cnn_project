# -*- coding: utf-8 -*-


class Word2VecTransformer(object):
    def __init__(self):
        self.dictionary = {}
        self.expand_symbol = u'</s>'
        self.vocabulary_size = None
        self.embeding_size = None

    def load_vectors(self, filename):
        with open(filename, 'r') as f:
            self.vocabulary_size, self.embeding_size = f.readline().decode('utf-8').strip().split(' ')
            for line in f:
                decode_line = line.decode('utf-8').strip()
                word = decode_line.split(' ')[0]
                vector = map(lambda x: float(x), decode_line.split(' ')[1:])
                self.dictionary[word] = vector

    def transform(self, lst):
        return [self.dictionary[word] for word in lst]

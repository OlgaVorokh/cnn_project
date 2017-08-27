# -*- coding: utf-8 -*-


class VocabularyWords(object):
    def __init__(self):
        self.words_index = {}
        self.words_emb = []
        self.embedding_size = None
        self.vocab_size = None
        self.input_cnt_words = None

    def build(self, filename):
        with open(filename, 'r') as f:
            self.vocab_size, self.embedding_size = f.readline().strip().split(' ')
            self.vocab_size = int(self.vocab_size)
            self.embedding_size = int(self.embedding_size)
            for index, line in enumerate(f.readlines()):
                parts = line.decode('utf-8').strip().split(' ')
                word = parts[0]
                self.words_index[word] = index
                self.words_emb.append(map(float, parts[1:]))

    def transform(self, words):
        self._check_input_cnt_words(words)

        result = []
        for word in words:
            if word is None:
                result.append(0)
                continue
            updated_word = word.lower()
            try:
                result.append(self.words_index[updated_word])
            except KeyError:
                result.append(0)

        return result

    def _check_input_cnt_words(self, words):
        if self.input_cnt_words is None:
            self.input_cnt_words = len(words)
        else:
            assert len(words) == self.input_cnt_words


class VocabularyTags(object):
    def __init__(self):
        self.tags_index = {}
        self.tags_size = None
        self.input_cnt_tags = None

    def build(self, tags_set):
        for key, value in enumerate(tags_set, 1):
            self.tags_index[value] = key
        self.tags_size = len(self.tags_index) + 1

    def transform(self, tags):
        self._check_input_cnt_tags(tags)

        result = []
        for tag in tags:
            result.append(0 if tag is None else self.tags_index[tag])
        return result

    def _check_input_cnt_tags(self, tags):
        if self.input_cnt_tags is None:
            self.input_cnt_tags = len(tags)
        else:
            assert len(tags) == self.input_cnt_tags


class VocabularyLabels(object):
    def __init__(self):
        self.labels_set = set()
        self.labels_index = {}
        self.labels_size = None
        self.input_cnt_labels = None

    def build(self, labels_set):
        for label in labels_set:
            if label is None:
                continue
            self.labels_set.add(self._normalize(label))
        spread_labels = []
        for label in self.labels_set:
            spread_labels.append(self._left(label))
            spread_labels.append(self._right(label))
        self.labels_set.add(u'shift')
        spread_labels.append(u'shift')

        for index, label in enumerate(spread_labels, 1):
            self.labels_index[label] = index

        self.labels_size = len(spread_labels)

    def _left(self, label):
        return u'left_{}'.format(label)

    def _right(self, label):
        return u'right_{}'.format(label)

    def _normalize(self, label):
        return label.split('_')[1]

    def transform(self, labels, for_y=False):
        if not for_y:
            self._check_input_cnt_labels(labels)
        result = []
        for label in labels:
            result.append(0 if label is None else self.labels_index[label])
        return result

    def _check_input_cnt_labels(self, labels):
        if self.input_cnt_labels is None:
            self.input_cnt_labels = len(labels)
        else:
            assert len(labels) == self.input_cnt_labels


class Vocabulary(object):
    def __init__(self, filename, tags, labels):
        self.vocabulary_labels = VocabularyLabels()
        self.vocabulary_tags = VocabularyTags()
        self.vocabulary_words = VocabularyWords()

        self.vocabulary_labels.build(labels_set=labels)
        self.vocabulary_words.build(filename=filename)
        self.vocabulary_tags.build(tags_set=tags)

    def transform(self, data_x, data_y):
        result_x = []
        for words, tags, labels in data_x:
            cur = []
            cur.extend(self.vocabulary_words.transform(words))
            cur.extend(self.vocabulary_tags.transform(tags))
            cur.extend(self.vocabulary_labels.transform(labels))
            result_x.append(cur)

        result_y = []
        labels_count = len(self.vocabulary_labels.labels_set) * 2 - 1
        for movement in data_y:
            movement_answer = [0] * labels_count
            index = self.vocabulary_labels.transform([movement], for_y=True)[0]
            movement_answer[index - 1] = 1
            result_y.append(movement_answer)

        return result_x, result_y





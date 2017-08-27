# -*- coding: utf-8 -*-

import numpy as np

import config

from . import vocabularies
from . import input_maker


def get_makers(configurations):
    makers = []
    for configuration in configurations:
        maker = input_maker.InputWordsFormatMaker(filename=configuration.input_filepath)
        maker.make()
        makers.append(maker)
    return makers


def get_tags_and_labels(makers):
    tags = set()
    labels = set()
    for maker in makers:
        tags.update(maker.tags_set)
        labels.update(maker.labels_set)
    return tags, labels


def load_data_and_labels():
    print 'Load data to makers'
    makers = get_makers([
        config.EnglishLiesTestConfig(),
        config.EnglishLiesTrainConfig(),
    ])

    print 'Get tags and labels from makers'
    tags, labels = get_tags_and_labels(makers)

    configuration_word2vec = config.Word2VecConfig()
    vocab = vocabularies.Vocabulary(
        filename=configuration_word2vec.vectors,
        tags=tags,
        labels=labels,
    )
    print 'Transform train data'
    x_train, y_train = vocab.transform(makers[1].train_x, makers[1].train_y)

    print 'Transform test data'
    x_test, y_test = vocab.transform(makers[0].train_x, makers[0].train_y)

    print 'Post data dict'
    return {
        'x_train': np.array(x_train),
        'y_train': np.array(y_train),
        'x_test': np.array(x_test),
        'y_test': np.array(y_test),
        'words_vocab_size': vocab.vocabulary_words.vocab_size,
        'tags_vocab_size': vocab.vocabulary_tags.tags_size,
        'labels_vocab_size': vocab.vocabulary_labels.labels_size,
        'input_cnt_words': vocab.vocabulary_words.input_cnt_words,
        'input_cnt_tags': vocab.vocabulary_tags.input_cnt_tags,
        'input_cnt_labels': vocab.vocabulary_labels.input_cnt_labels,
        'embeddings': np.array(vocab.vocabulary_words.words_emb),
    }


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

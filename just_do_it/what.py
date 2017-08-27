# -*- coding: utf-8 -*-


import numpy as np
import codecs
import re
import itertools
from collections import Counter

DATA_TRAIN_PATH = './data/ru_syntagrus-ud-train.conllu'
DATA_TEST_PART = './data/ru_syntagrus-ud-test.conllu'
DATA_PATH = None

# Constants for the column indices
COLCOUNT = 10
ID, FORM, LEMMA, UPOSTAG, XPOSTAG, FEATS, HEAD, DEPREL, DEPS, MISC = range(COLCOUNT)
COLNAMES = u"ID,FORM,LEMMA,UPOSTAG,XPOSTAG,FEATS,HEAD,DEPREL,DEPS,MISC".split(u",")

# If it's nessesary to expand a sentence, it makes by adding this fake word
FAKE_WORD = 'aaaaaaaaaaa'

WORD2VEC_VECTOR_SIZE = 40

MARK_SUBJECT = 1
MARK_PREDICATE = 2
MARK_ENOTHER = 0
NUM_CLASSES = 3


def tree_generator(inp):
    word_lines = []  # List of token/word lines of the current sentence
    for line in inp:
        line = line.rstrip(u"\n")
        if not line and word_lines:  # Sentence done
            yield word_lines
            word_lines = []
        if not line or line[0] == u"#":
            continue
        if line[0].isdigit():
            cols = line.split("\t")
            word_lines.append(cols)
    else:  # end of file
        if word_lines:
            yield word_lines


def get_words_from_tree(tree):
    words = []
    for line in tree:
        if line[UPOSTAG] == 'POINT':
            words.append(u'препинание')
            continue
        word = line[FORM].lower()
        words.append(word)

    return words


def make_sentence_graph(tree):
    g = [dict() for _ in xrange(len(tree) + 1)]
    for index, line in enumerate(tree):
        ver_from = int(line[HEAD])
        edge_how = line[DEPREL]
        g[ver_from][edge_how] = index + 1
    return g


# =================================== REWRITE ALL ABOVE ============================================
# sent_id = 2
# text = About ANSI SQL query mode
# ID, FORM, LEMMA, UPOSTAG, XPOSTAG, FEATS, HEAD, DEPREL, DEPS, MISC = range(COLCOUNT)
# 1	  About	   _	ADP	       _	   _	 5	   case	   _	 _
# 2	ANSI	_	PROPN	SG-NOM	_	5	compound	_	_
# 3	SQL	_	PROPN	SG-NOM	_	2	flat	_	_
# 4	query	_	NOUN	SG-NOM	_	2	flat	_	_
# 5	mode	_	NOUN	_	_	0	root	_	_


def get_answer_from_tree(ver, edge, prev, g, tree, answer):
    ver_edges = g[ver]

    if edge == 'root':
        if 'nsubj' in ver_edges or 'nsubjpass' in ver_edges:
            answer[ver - 1] = MARK_PREDICATE
        else:
            ver_info = tree[ver - 1]
            if ver_info[LEMMA] == 'NOUN' or ver_info[LEMMA] == 'PRON':
                answer[ver - 1] = MARK_SUBJECT
            else:
                answer[ver - 1] = MARK_PREDICATE
    elif edge.startswith('nsubj'):
        if 'nsubj' in ver_edges or 'nsubjpass' in ver_edges:
            answer[ver - 1] = MARK_PREDICATE
        else:
            answer[ver - 1] = MARK_SUBJECT
    elif edge == 'xcomp' or edge.startswith('aux') or edge.startswith('conj'):
        answer[ver - 1] = answer[prev - 1]
    elif edge.startswith('acl') or edge.startswith('advcl'):
        if 'nsubj' in ver_edges or 'nsubjpass' in ver_edges:
            answer[ver - 1] = MARK_PREDICATE
        else:
            ver_info = tree[ver - 1]
            if ver_info[LEMMA] == 'VERB':
                verb_info = dict(_.split('=') for _ in ver_info[FEATS].split('|'))
                if not (ver_info['VerbForm'] == 'Trans' or ver_info['VerbForm'] == 'Part'):
                    answer[ver - 1] = MARK_PREDICATE

    for e in ver_edges:
        get_answer_from_tree(ver_edges[e], e, ver, g, tree, answer)


def data_generator(inp):
    for tree in tree_generator(inp):
        g = make_sentence_graph(tree)
        words = get_words_from_tree(tree)
        answer = [MARK_ENOTHER] * len(words)
        get_answer_from_tree(0, 'start', -1, g, tree, answer)
        yield words, answer


def load_data_and_labels():
    print ('Load data from file...')
    with codecs.getreader("utf-8")(open(DATA_TRAIN_PATH, mode='U')) as inp:
        sentences = []
        labels = []
        for words, answers in data_generator(inp):
            sentences.append(words)
            labels.append(answers)

    return sentences, labels


# Updaters
# ==================================================

def updater_all_sentence(x, y, len_seq):
    x_update = []
    y_update = list(y)
    for index in xrange(len(x)):
        x_update.append(' '.join(x[index]))
        while len(y_update[index]) > len_seq:
            y_update[index].pop()
        y_update[index].extend([0] * max(0, len_seq - len(y[index])))
    return x_update, y_update


def updater_both_k_words(x, y, k=3):
    x_update = []
    y_update = []
    for i in xrange(len(x)):
        for j in xrange(len(x[i])):
            for k in xrange(- (k / 2), k / 2 + 1):
                new_expample = []
                if j + k < 0 or j + k >= len(x[i]):
                    new_expample.append(FAKE_WORD)
                else:
                    new_expample.append(x[i][j + k])

            x_update.append(' '.join(new_expample))
            y_update.append([0.] * NUM_CLASSES)
            y_update[-1][y[i][j]] = 1.
    return x_update, y_update


def updater_both_pairs(x, y):
    x_update = []
    y_update = []
    for num_seq in xrange(len(x)):
        seq = x[num_seq]
        label_seq = y[num_seq]
        for i in xrange(len(seq)):
            for j in xrange(i, len(seq)):
                x_update.append(' '.join([seq[i], seq[j]]))

                label_first = label_seq[i]
                label_second = label_seq[j]
                y_update.append([0.] * NUM_CLASSES)
                if not label_first and not label_second:
                    y_update[-1][0] = 1.
                if (not label_first and label_second == 1) or (label_first == 1 and not label_second):
                    y_update[-1][1] = 1.
                if (not label_first and label_second == 2) or (label_first == 2 and not label_second):
                    y_update[-1][2] = 1.
                if (label_first == 1 and label_second == 2) or (label_first == 2 and label_second == 1):
                    y_update[-1][3] = 1.
                if (label_first == 1 and label_second == 1):
                    y_update[-1][4] = 1.
                if (label_first == 2 and label_second == 2):
                    y_update[-1][5] = 1.
    return x_update, y_update


def updater_main_pairs(x, y):
    x_update = []
    y_update = []
    for num in xrange(len(x)):
        seq = x[num]
        for i in xrange(len(seq)):
            for j in xrange(i, len(seq)):
                x_update.append(' '.join([seq[i], seq[j]]))

                label_first = y[num][i]
                label_second = y[num][j]
                y_update.append([0.] * NUM_CLASSES)
                if not label_first and not label_second:
                    y_update[-1][0] = 1.
                if not label_first and label_second:
                    y_update[-1][1] = 1.
                if label_first and not label_second:
                    y_update[-1][2] = 1.
                if label_first and label_second:
                    y_update[-1][3] = 1.
    return x_update, y_update


# ==================================================

def get_data(data_format, len_seq=None, status='TRAIN'):
    ''' Return data in some user format. '''
    global MARK_ENOTHER
    global MARK_SUBJECT
    global MARK_PREDICATE
    global NUM_CLASSES
    global DATA_PATH

    if status == 'TRAIN':
        DATA_PATH = DATA_TRAIN_PATH
    else:
        DATA_PATH = DATA_TEST_PART

    if data_format == 'ALL_SENTENCE':
        MARK_ENOTHER = 0
        MARK_SUBJECT = 1
        MARK_PREDICATE = 1
        x, y = load_data_and_labels()
        x_update, y_update = updater_all_sentence(x, y, len_seq)

    if data_format == 'BOTH_THREE_WORDS':  # 0.845 полчаса
        MARK_ENOTHER = 0
        MARK_PREDICATE = 2
        MARK_SUBJECT = 1
        NUM_CLASSES = 3
        x, y = load_data_and_labels()
        x_update, y_update = updater_both_k_words(x, y, k=3)

    if data_format == 'MAIN_THREE_WORDS':  # 0.848
        MARK_ENOTHER = 0
        MARK_SUBJECT = 1
        MARK_PREDICATE = 1
        NUM_CLASSES = 2
        x, y = load_data_and_labels()
        x_update, y_update = updater_both_k_words(x, y, k=3)

    if data_format == 'BOTH_FIVE_WORDS':  # 0.838
        MARK_ENOTHER = 0
        MARK_SUBJECT = 1
        MARK_PREDICATE = 2
        NUM_CLASSES = 3
        x, y = load_data_and_labels()
        x_update, y_update = updater_both_k_words(x, y, k=5)

    if data_format == 'MAIN_FIVE_WORDS':  # 0.845
        MARK_ENOTHER = 0
        MARK_SUBJECT = 1
        MARK_PREDICATE = 1
        NUM_CLASSES = 2
        x, y = load_data_and_labels()
        x_update, y_update = updater_both_k_words(x, y, k=5)

    if data_format == 'MAIN_SEVEN_WORDS':  # 0.845
        MARK_ENOTHER = 0
        MARK_SUBJECT = 1
        MARK_PREDICATE = 1
        NUM_CLASSES = 2
        x, y = load_data_and_labels()
        x_update, y_update = updater_both_k_words(x, y, k=7)

    if data_format == 'BOTH_PAIRS':  # 0.827
        MARK_ENOTHER = 0
        MARK_PREDICATE = 2
        MARK_SUBJECT = 1
        NUM_CLASSES = 6
        x, y = load_data_and_labels()
        x_update, y_update = updater_both_pairs(x, y)

    if data_format == 'MAIN_PAIRS':  # 0.827
        MARK_ENOTHER = 0
        MARK_PREDICATE = 1
        MARK_SUBJECT = 1
        NUM_CLASSES = 4
        x, y = load_data_and_labels()
        x_update, y_update = updater_main_pairs(x, y)

    if data_format == 'olyaolya':  # 0.845
        MARK_ENOTHER = 0
        MARK_SUBJECT = 1
        MARK_PREDICATE = 1
        NUM_CLASSES = 2
        x, y = load_data_and_labels()
        x_update, y_update = updater_both_k_words(x, y, k=9)

    del x
    del y
    return x_update, y_update


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


if __name__ == '__main__':
    x, y = load_data_and_labels()
    count_predicate = 0
    count_subject = 0.
    count_all = 0
    for line in y:
        count_predicate += line.count(MARK_PREDICATE)
        count_subject += line.count(MARK_SUBJECT)
        count_all += len(line)

    print 'PREDICATE:', count_predicate
    print 'SUBJECT:', count_subject
    print 'Diff:', count_predicate + count_subject, '/', count_all
    print '%', float(count_subject + count_predicate) / count_all
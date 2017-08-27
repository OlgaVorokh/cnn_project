# -*- coding: utf-8 -*-

from .helper_sentense import SentenceGetter, SentenceParser


class InputWordsFormatMaker(object):
    def __init__(self, filename):
        self.filename = filename

        self.tags_set = set()
        self.labels_set = set()

        self.train_x = []
        self.train_y = []

    def make(self):
        sentence_getter = SentenceGetter(filename=self.filename)
        sentence_generator = sentence_getter()

        for sentence_parser in sentence_generator:
            graph_builder = SentenceGraphBuilder()
            graph_builder.build(sentence_parser=sentence_parser)

            punctuation_remover = PunctuationRemover(sentence_parser=sentence_parser, graph_builder=graph_builder)
            update_sp, update_g = punctuation_remover.update()
            recurser = Recurser(sentence_parser=update_sp, graph_builder=update_g)

            try:
                # this exception wrapper is made because sometimes parser structure in input file is not correct
                recurser.calculate()
            except:
                pass

            self._save_recurser_state(recurser=recurser)

    def _save_recurser_state(self, recurser):
        self.train_x.extend(recurser.train_x)
        self.train_y.extend(recurser.train_y)

        self.tags_set.update(recurser.tags_set)
        self.labels_set.update(recurser.labels_set)


class PunctuationRemover(object):
    def __init__(self, sentence_parser, graph_builder):
        self.sentence_parser = sentence_parser
        self.graph = graph_builder.graph
        self.reversed_graph = graph_builder.reversed_graph

        self._new_indexes = {}
        self._lines = self.sentence_parser.lines

    def update(self):
        self._mark_deleted_lines(0)

        pos = 1
        for index in xrange(len(self.sentence_parser)):
            if self.sentence_parser[index].head == u'None':
                self._new_indexes[index + 1] = None
                continue
            self._new_indexes[index + 1] = pos
            pos += 1

        self._update_indexes(0)

        new_lines = []
        for index in xrange(len(self.sentence_parser)):
            if self._new_indexes[index + 1] is not None:
                new_lines.append(self.sentence_parser.lines[index])

        update_sentence_parser = SentenceParser(lines=new_lines)
        graph_builder = SentenceGraphBuilder()
        graph_builder.build(sentence_parser=update_sentence_parser)
        return update_sentence_parser, graph_builder

    def _mark_deleted_lines(self, vertex):
        if vertex == 0:
            for next in self.graph[vertex]:
                self._mark_deleted_lines(next)
            return

        line_parser = self.sentence_parser[vertex - 1]
        mark = line_parser.upostag == u'PUNCT' or line_parser.head == u'None'
        if mark:
            line_parser.set_head(None)
            self.sentence_parser.lines[vertex - 1] = line_parser.line
        for next_ver in self.graph[vertex]:
            if mark:
                line_parser = self.sentence_parser[next_ver - 1]
                line_parser.set_head(None)
                self.sentence_parser.lines[next_ver - 1] = line_parser.line
            self._mark_deleted_lines(next_ver)

    def _update_indexes(self, vertex):
        if vertex == 0:
            for next in self.graph[vertex]:
                self._update_indexes(next)
            return

        new_number = self._new_indexes[vertex]
        if new_number is None:
            return

        line_parser = self.sentence_parser[vertex - 1]
        line_parser.set_id(new_number)
        self.sentence_parser.lines[vertex - 1] = line_parser.line
        for next in self.graph[vertex]:
            line_parser = self.sentence_parser[next - 1]
            line_parser.set_head(new_number)
            self.sentence_parser.lines[next - 1] = line_parser.line
            self._update_indexes(next)


class SentenceGraphBuilder(object):
    def __init__(self):
        self.graph = None
        self.reversed_graph = None

    def build(self, sentence_parser):
        self._build_graph(sentence_parser)
        self._build_reversed_graph(sentence_parser)

    def _build_graph(self, sentence_parser):
        self.graph = [dict() for _ in xrange(len(sentence_parser) + 1)]
        for index in xrange(len(sentence_parser)):
            line_parser = sentence_parser[index]
            from_ver = int(line_parser.head)
            by_edge = line_parser.deprel
            self.graph[from_ver][index + 1] = by_edge

    def _build_reversed_graph(self, sentence_parser):
        self.reversed_graph = [[] for _ in xrange(len(sentence_parser) + 1)]
        for index in xrange(1, len(self.graph)):
            for ver in self.graph[index]:
                self.reversed_graph[ver].append(index)


class Recurser(object):
    def __init__(self, sentence_parser, graph_builder):
        self.sentence_parser = sentence_parser
        self.graph = graph_builder.graph
        self.reversed_graph = graph_builder.reversed_graph
        self.links = [0] * len(self.graph)

        self.tags_set = set()
        self.labels_set = set()

        self.train_x = []
        self.train_y = []

        # for vertex we have dict with fields 'left' and 'right', that means vertexes,
        # that are situated on the left or right side from word with number vertex
        self.rec_graph = [{'left': [], 'right': []} for _ in xrange(len(self.graph))]
        # index start from 1, because this is the number of the first word according
        # to the sentence graph (variable self.graph) structure
        self.index = 1
        # keep indexes of words; the first word have index 1, not zero!
        self.stack = []

    def calculate(self):
        self._build_links()
        self._run()

    def _build_links(self):
        for index in xrange(1, len(self.graph)):
            self.links[index] += len(self.graph[index])

    def _run(self):
        if self.index == len(self.graph) and len(self.stack) <= 1:
            return

        if len(self.stack) < 2 or self.links[self.stack[-1]] and self.links[self.stack[-2]]:
            self._save_state(self._get_shift())
            # self._print_last_state()
            self.stack.append(self.index)
            self.index += 1
            self._run()
            return

        left_direction = self._get_arc(u'left', self.stack[-1], self.stack[-2])
        right_direction = self._get_arc(u'right', self.stack[-2], self.stack[-1])
        if left_direction is None and right_direction is None:
            self._save_state(self._get_shift())
            # self._print_last_state()
            self.stack.append(self.index)
            self.index += 1
            self._run()
            return

        if left_direction is not None:
            self._save_state(left_direction)
            # self._print_last_state()
            self.rec_graph[self.stack[-1]]['left'].append(self.stack[-2])
            self._delete_links_for_pop_ver(self.stack.pop(-2))
            self._run()
            return

        if right_direction is not None:
            self._save_state(right_direction)
            # self._print_last_state()
            self.rec_graph[self.stack[-2]]['right'].append(self.stack[-1])
            self._delete_links_for_pop_ver(self.stack.pop(-1))
            self._run()
            return

    def _delete_links_for_pop_ver(self, pop_ver):
        for ver in self.reversed_graph[pop_ver]:
            self.links[ver] -= 1

    def _print_last_state(self):
        w, t, l = self.train_x[-1]
        result = self.train_y[-1]
        print 'index', self.index
        print 'words', w
        print 'tags', t
        print 'labels', l
        print 'result', result
        print
        print

    def _get_from(self, array, index):
        if 0 <= index < len(array):
            return array[index]
        return None

    def _get_label_from_graph(self, ver_from, ver_to):
        if ver_from is None or ver_to is None:
            return None
        if ver_to not in self.graph[ver_from]:
            return None
        return self.graph[ver_from][ver_to]

    def _get_arc(self, direction, ver_to, ver_from):
        edge = self._get_label_from_graph(ver_to, ver_from)
        return edge if edge is None else u'_'.join([direction, edge])

    def _get_shift(self):
        return u'shift'

    def _save_state(self, result_move):
        words_indexes = self._get_words_indexes()

        words = self._convert_indexes_to_words(words_indexes)
        tags = self._get_tags(words_indexes)
        labels = self._get_labels(words_indexes)

        self.tags_set.update(set(tags))
        self.labels_set.update(set(labels))

        self.train_x.append([words, tags, labels])
        self.train_y.append(result_move)

    def _get_words_indexes(self):
        # we need to make -1 in future for all indexes, because at first we save all of it
        # from syntax graph and in graph zero index is ROOT, not the first word
        words_indexes = []

        # put top3 word indexes from stack
        stack_len = len(self.stack)
        words_indexes.extend([
            self._get_from(self.stack, stack_len - 1),
            self._get_from(self.stack, stack_len - 2),
            self._get_from(self.stack, stack_len - 3),
        ])

        # put top3 word indexes from buffer
        words_indexes.extend([
            self.index if self.index - 1 < len(self.sentence_parser) else None,
            self.index + 1 if self.index < len(self.sentence_parser) else None,
            self.index + 2 if self.index + 1 < len(self.sentence_parser) else None,
        ])

        # put leftmost and rightmost children for top1 stack index-word
        left_most_1, left_most_2 = self._get_left_most(self.stack, stack_len - 1)
        right_most_1, right_most_2 = self._get_right_most(self.stack, stack_len - 1)
        words_indexes.extend([
            left_most_1, right_most_1,
            left_most_2, right_most_2,
        ])

        # put left most and right most children for top2 stack index-word
        left_most_1, left_most_2 = self._get_left_most(self.stack, stack_len - 2)
        right_most_1, right_most_2 = self._get_right_most(self.stack, stack_len - 2)
        words_indexes.extend([
            left_most_1, right_most_1,
            left_most_2, right_most_2,
        ])

        # put left left most and right right most children of top1 stack index-word
        left_left_most = self._get_left_left_most(self.stack, stack_len - 1)
        right_right_most = self._get_right_right_most(self.stack, stack_len - 1)
        words_indexes.extend([
            left_left_most, right_right_most,
        ])

        # put left left most and right right most children of top2 stack index-word
        left_left_most = self._get_left_left_most(self.stack, stack_len - 2)
        right_right_most = self._get_right_right_most(self.stack, stack_len - 2)
        words_indexes.extend([
            left_left_most, right_right_most,
        ])

        return words_indexes

    def _get_left_most(self, array, index):
        ver = self._get_from(array, index)
        if ver is None:
            return None, None
        lst = self.rec_graph[ver]['left']
        lst.sort()
        left_most_1 = lst[0] if len(lst) else None
        left_most_2 = lst[1] if len(lst) > 1 else None
        return left_most_1, left_most_2

    def _get_right_most(self, array, index):
        ver = self._get_from(array, index)
        if ver is None:
            return None, None
        lst = self.rec_graph[ver]['right']
        lst.sort()
        right_most_1 = lst[-1] if len(lst) else None
        right_most_2 = lst[-2] if len(lst) > 1 else None
        return right_most_1, right_most_2

    def _get_left_left_most(self, array, index):
        left_most_1, left_nost_2 = self._get_left_most(array, index)
        if left_most_1 is None:
            return None
        lst = self.rec_graph[left_most_1]['left']
        lst.sort()
        return lst[0] if len(lst) else None

    def _get_right_right_most(self, array, index):
        right_most_1, right_most_2 = self._get_right_most(array, index)
        if right_most_1 is None:
            return None
        lst = self.rec_graph[right_most_1]['right']
        lst.sort()
        return lst[-1] if len(lst) else None

    def _convert_indexes_to_words(self, indexes):
        result = []
        for index in indexes:
            result.append(index if index is None else self.sentence_parser[index - 1].form)
        return result

    def _get_tags(self, indexes):
        result = []
        for index in indexes:
            result.append(index if index is None else self.sentence_parser[index - 1].upostag)
        return result

    def _get_labels(self, words_indexes):
        # 0 - stack top 1
        # 1 - stack top 2
        # 2 - stack top 3

        # 3 - buffer top 1
        # 4 - buffer top 2
        # 5 - buffer top 3

        # 6 - left first most of 0
        # 7 - right first most of 0
        # 8 - left second most of 0
        # 9 - right second most of 0

        # 10 - left first most of 1
        # 11 - right first most of 1
        # 12 - left second most of 1
        # 13 - right second most of 1

        # 14 - left left  most of 0
        # 15 - right right most of 0

        # 16 - left left most of 1
        # 17 - right right most of 1
        labels = []

        labels.extend([
            self._get_arc(u'left', words_indexes[0], words_indexes[6]),
            self._get_arc(u'right', words_indexes[0], words_indexes[7]),
            self._get_arc(u'left', words_indexes[0], words_indexes[8]),
            self._get_arc(u'right', words_indexes[0], words_indexes[9]),
        ])

        labels.extend([
            self._get_arc(u'left', words_indexes[1], words_indexes[10]),
            self._get_arc(u'right', words_indexes[1], words_indexes[11]),
            self._get_arc(u'left', words_indexes[1], words_indexes[12]),
            self._get_arc(u'right', words_indexes[1], words_indexes[13]),
        ])

        labels.extend([
            self._get_arc(u'left', words_indexes[6], words_indexes[14]),
            self._get_arc(u'right', words_indexes[7], words_indexes[15]),
        ])

        labels.extend([
            self._get_arc(u'left', words_indexes[10], words_indexes[16]),
            self._get_arc(u'right', words_indexes[11], words_indexes[17]),
        ])

        return labels

# -*- coding: utf-8 -*-


class SentenceGetter(object):
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, ):
        sentence_lines = []
        with open(self.filename, 'r') as f:
            for line in f:
                decode_line = line.decode('utf-8').strip()

                if not decode_line:
                    if sentence_lines:
                        yield SentenceParser(lines=sentence_lines)
                        sentence_lines = []
                    continue

                if decode_line[0] == u'#':
                    continue

                if decode_line[0].isdigit():
                    sentence_lines.append(decode_line)

            if sentence_lines:
                yield SentenceParser(lines=sentence_lines)


class SentenceParser(object):
    def __init__(self, lines):
        self.lines = lines

    def __getitem__(self, index):
        assert isinstance(index, int)
        assert 0 <= index <= len(self.lines)
        return SentenceLineParser(line=self.lines[index])

    def __len__(self):
        return len(self.lines)


class SentenceLineParser(object):
    def __init__(self, line):
        self.columns = line.split(u'\t')
        self.structure_names = u'ID,FORM,LEMMA,UPOSTAG,XPOSTAG,FEATS,HEAD,DEPREL,DEPS,MISC'

    @property
    def structure(self):
        return dict((value, index) for index, value in enumerate(self.structure_names.split(u',')))

    @property
    def id(self):
        index = self.structure[u'ID']
        return self.columns[index]

    def set_id(self, new_id):
        index = self.structure[u'ID']
        self.columns[index] = unicode(new_id)

    @property
    def form(self):
        index = self.structure[u'FORM']
        return self.columns[index]

    @property
    def lemma(self):
        index = self.structure[u'LEMMA']
        return self.columns[index]

    @property
    def upostag(self):
        index = self.structure[u'UPOSTAG']
        return self.columns[index]

    @property
    def xpostag(self):
        index = self.structure[u'XPOSTAG']
        return self.columns[index]

    @property
    def feats(self):
        index = self.structure[u'FEATS']
        return self.columns[index]

    @property
    def head(self):
        index = self.structure[u'HEAD']
        return self.columns[index]

    def set_head(self, new_head):
        index = self.structure[u'HEAD']
        self.columns[index] = unicode(new_head)

    @property
    def deprel(self):
        index = self.structure[u'DEPREL']
        return self.columns[index]

    @property
    def deps(self):
        index = self.structure[u'DEPS']
        return self.columns[index]

    @property
    def misc(self):
        index = self.structure[u'MISC']
        return self.columns[index]

    @property
    def line(self):
        return u'\t'.join(self.columns)

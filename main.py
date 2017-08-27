# -*- coding: utf-8 -*-

from just_do_it import prestart_data_creator


def main():
    creator = prestart_data_creator.PreStartDataCreator()
    creator.make_word2vec_input()

if __name__ == '__main__':
    main()
